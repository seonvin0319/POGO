# File: pogogo/run_comparison.py

#!/usr/bin/env python3
"""
POGO 통합 학습 실험 런처 (순차 실행 버전)
- config.yaml에 정의된 환경/하이퍼를 순차 수행 (병렬처리 없음)
- 통합 학습: actor_one과 actor_two를 동시에 학습
- TD target에 online policy 사용
- 체크포인트에서 이어서 학습 가능 (load 모드)
- GPU 사용
"""

import os
import sys
import time
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from typing import Optional, List, Dict, Tuple

# ----------------------------
# 유틸
# ----------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def safe(s: str) -> str:
    return s.replace('/', '_').replace('-', '_')

def load_yaml(path: Path):
    with path.open('r') as f:
        return yaml.safe_load(f)

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def tail(path: Path, n=50):
    if not path.exists():
        return ""
    with path.open('r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return ''.join(lines[-n:])

def format_w2_weight(w: float) -> str:
    """w2 weight를 적응형으로 포맷팅 (필요한 소수점 자리수만 표시)"""
    # 정수인 경우 그대로 반환
    if abs(w - round(w)) < 1e-10:
        return str(int(round(w)))
    
    # 소수점이 있는 경우, 최대 6자리까지 시도하면서 trailing zero 제거
    # 각 precision에서 포맷팅 후 원래 값과 비교
    for precision in range(1, 7):
        formatted = f"{w:.{precision}f}"
        # trailing zero와 소수점 제거
        cleaned = formatted.rstrip('0').rstrip('.')
        # 포맷팅된 값이 원래 값과 같으면 반환
        try:
            if abs(float(cleaned) - w) < 1e-10:
                return cleaned
        except ValueError:
            continue
    
    # 위 방법이 실패하면 최대 precision으로 표시
    s = f"{w:.6f}".rstrip('0').rstrip('.')
    return s if s else "0"

# ----------------------------
# 체크 / 실행 함수들
# ----------------------------
def find_checkpoint(ckpt_dir: Path, step: Optional[int] = None) -> Optional[Path]:
    """체크포인트 찾기. step이 지정되면 _mid_<step>_*_actor_0 파일을 찾음."""
    if step is not None:
        # 정확 매치
        for f in ckpt_dir.glob(f"*_mid_{step}_*_actor_0"):
            return f
        # 근접 매치
        best = None
        best_diff = 1e18
        for f in ckpt_dir.glob("*_mid_*_actor_0"):
            parts = f.stem.split('_')
            for i, p in enumerate(parts):
                if p == 'mid' and i + 1 < len(parts):
                    try:
                        t = int(parts[i + 1])
                    except Exception:
                        continue
                    diff = abs(t - step)
                    if diff < best_diff:
                        best_diff = diff
                        best = f
        return best
    else:
        # 가장 최근 체크포인트 찾기 (actor_0 기준)
        checkpoints = list(ckpt_dir.glob("*_actor_0"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

def training_done(log_file: Path) -> bool:
    """학습이 완료되었는지 로그로 판정(최종 표기 문자열 기반)."""
    if not log_file.exists():
        return False
    txt = log_file.read_text(errors='replace')
    return ('======== Final Evaluation' in txt
            and '[FINAL] Deterministic:' in txt
            and '[FINAL] Stochastic:' in txt)


def run_phase(
    pyexec: Path, root_dir: Path, args: List[str], log_path: Path, env: Optional[Dict] = None
) -> Tuple[int, Optional[str]]:
    """main.py 한 번 실행. rc, 예외메시지 반환.
    예시 코드처럼 단순하게 파일에 직접 리다이렉트합니다.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rc, err = 0, None
    
    try:
        env_vars = (env or os.environ.copy())
        env_vars['PYTHONUNBUFFERED'] = '1'
        
        with log_path.open('w', encoding='utf-8') as logf:
            proc = subprocess.Popen(
                [str(pyexec), '-u', 'main.py'] + args,
                cwd=str(root_dir),
                env=env_vars,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True
            )
            proc.wait()
            rc = proc.returncode
    except Exception as e:
        rc, err = -999, f"{type(e).__name__}: {e}"
    return rc, err

def strip_suffix(load_prefix: str) -> str:
    """..._actor_i / _critic 등의 접미를 제거한 프리픽스 반환."""
    suffixes = [
        '_actor_0', '_actor_1', '_actor_2',  # multi-actor용
        '_actor', '_critic', '_behavior',
        '_actor_optimizer', '_critic_optimizer', '_behavior_optimizer',
    ]
    for suf in suffixes:
        if load_prefix.endswith(suf):
            return load_prefix[:-len(suf)]
    return load_prefix

# ----------------------------
# 메인 파이프라인
# ----------------------------
def run_unified_training(env_id: str, seed: int, w2_weights: List[float], 
                         lr: float, max_steps: int, eval_freq: int, split_ratio: float,
                         root_dir: Path, pyexec: Path) -> dict:
    """통합 학습 실험: 0 → max_steps (모든 actor 동시 학습)"""
    start = time.time()
    split_step = int(round(max_steps * split_ratio))
    
    # 로그/체크포인트 디렉토리
    logs_root = Path('logs')
    w2_str = "_".join([format_w2_weight(w) for w in w2_weights])
    base = logs_root / safe(env_id) / f"w2_{w2_str}" / f"seed_{seed}"
    ckpt_dir = base / "checkpoints"
    log_dir = base / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 완료된 로그 확인
    existing_logs = list(log_dir.glob("POGO_unifiedd*.log"))
    for log_file in existing_logs:
        if training_done(log_file):
            print(f"⏭️  통합 학습 스킵: {env_id} seed={seed} — 이미 완료됨 ({log_file.name})")
            return {
                'env': env_id, 'seed': seed, 'experiment_type': 'unified',
                'status': 'skipped_already_done', 'duration_min': 0.0,
                'log': str(log_file.resolve()), 'checkpoint_dir': str(ckpt_dir.resolve())
            }
    
    
    print(f"🔄 통합 학습 시작: {env_id} seed={seed} — 0→{max_steps}")
    
    log_file = log_dir / f"POGO_unified_{safe(env_id)}_{seed}_{now_str().replace(':','-')}.log"
    
    env_vars = os.environ.copy()
    env_vars['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 사용
    
    args_list = [
        '--env', env_id,
        '--seed', str(seed),
        '--max_timesteps', str(max_steps),
        '--eval_freq', str(eval_freq),
        '--w2_weights'] + [str(w) for w in w2_weights] + [
        '--lr', str(lr),
        '--checkpoint_dir', str(ckpt_dir),
        '--save_model',
    ]
    
    rc, err = run_phase(
        pyexec, root_dir,
        args=args_list,
        log_path=log_file,
        env=env_vars
    )
    
    if rc != 0 or err:
        print(f"❌ 통합 학습 실패: rc={rc}, err={err}\n{tail(log_file, 30)}")
        return {
            'env': env_id, 'seed': seed, 'experiment_type': 'unified',
            'status': 'failed', 'rc': rc, 'err': err, 'log': str(log_file.resolve())
        }
    
    dur_min = (time.time() - start) / 60.0
    return {
        'env': env_id, 'seed': seed, 'experiment_type': 'unified',
        'status': 'success', 'duration_min': round(dur_min, 3),
        'log': str(log_file.resolve()), 'checkpoint_dir': str(ckpt_dir.resolve())
    }


def main():
    ap = ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--root_dir', default='.')
    ap.add_argument('--pyexec', default='/home/offrl/miniconda3/envs/offrl/bin/python')
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    pyexec = Path(args.pyexec)
    # config.yaml 경로: 현재 디렉토리에서 먼저 찾고, 없으면 root_dir 기준으로 찾기
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # 상대 경로인 경우, 현재 디렉토리에서 먼저 찾기
        if config_path.exists():
            config_path = config_path.resolve()
        elif (root_dir / config_path).exists():
            config_path = root_dir / config_path
        elif (root_dir.parent / config_path).exists():
            config_path = root_dir.parent / config_path
        else:
            # 현재 디렉토리를 기본으로 사용
            config_path = Path.cwd() / config_path
    cfg = load_yaml(config_path)

    common = cfg['common']
    max_steps  = common['max_timesteps']
    eval_freq  = common['eval_freq']
    seeds      = common['seeds']
    split_ratio= common.get('split_ratio', 0.5)

    # 환경 순서 정의: halfcheetah → hopper → walker2d → antmaze
    env_order = {
        'halfcheetah': ['medium', 'medium-replay', 'medium-expert'],
        'hopper': ['medium', 'medium-replay', 'medium-expert'], 
        'walker2d': ['medium', 'medium-replay', 'medium-expert'], 
        'antmaze': ['umaze-v2', 'umaze-diverse-v2', 'medium-play-v2', 'medium-diverse-v2', 'large-play-v2', 'large-diverse-v2'], 
    }

    all_runs = []
    for env_key in env_order.keys():
        if env_key not in cfg['environments']:
            continue
        datasets = cfg['environments'][env_key]
        for dataset_key in env_order[env_key]:
            if dataset_key not in datasets:
                continue
            env_cfg = datasets[dataset_key]
            env_id = f"{env_key}-{dataset_key}"
            
            all_runs.append({
                'env_id': env_id,
                'w2_weights': env_cfg['w2_weights'],
                'lr': env_cfg['learning_rate'],
            })

    print(f"🔬 총 {len(all_runs)*len(seeds)}개 실험 예정 (통합 학습)")
    print(f"📋 순차 실행 모드")
    print(f"🔄 통합 학습: GPU 사용 (online policy for TD target)")
    
    results = []
    t0 = time.time()

    for seed in seeds:
        print(f"\n🎲 SEED {seed} 시작")
        for e in all_runs:
            w2_str = ", ".join([format_w2_weight(w) for w in e['w2_weights']])
            print(f"— {e['env_id']} | w2_weights=[{w2_str}] lr={e['lr']}")
            
            # 통합 학습 실행
            print(f"  🔄 통합 학습 실행 중...")
            r = run_unified_training(
                env_id=e['env_id'], seed=seed,
                w2_weights=e['w2_weights'],
                lr=e['lr'],
                max_steps=max_steps, eval_freq=eval_freq,
                split_ratio=split_ratio,
                root_dir=root_dir, pyexec=pyexec
            )
            results.append(r)
            print(f"  ✅ 통합 학습 완료: {r['status']}")
            
            print(f"✅ {e['env_id']} seed={seed} 완료")

    # 결과 저장
    ts = now_str().replace(':','-')
    out_dir = Path(f"results_{ts}")
    out_dir.mkdir(exist_ok=True)
    
    # JSON 저장
    write_json(out_dir / "unified_summary.json", results)

    # CSV 저장
    csv_file = out_dir / "unified_results.csv"
    with csv_file.open('w', encoding='utf-8') as f:
        f.write("env,seed,experiment_type,status,rc,err,duration_min,log,checkpoint_dir\n")
        for r in results:
            f.write(f"{r.get('env','')},{r.get('seed','')},{r.get('experiment_type','')},"
                    f"{r.get('status','')},{r.get('rc','')},{r.get('err','')},"
                    f"{r.get('duration_min','')},{r.get('log','')},{r.get('checkpoint_dir','')}\n")

    mins = (time.time() - t0) / 60.0
    print("\n🏁 완료 | 총 소요 {:.1f}분 | 결과: {}".format(mins, out_dir))
    print(f"📊 통합 학습 결과: {csv_file}")

if __name__ == "__main__":
    main()

