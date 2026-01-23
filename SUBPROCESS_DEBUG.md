# SubprocVecEnv EOFError - Diagnosis and Fixes

## Changes Made

### 1. Added `refresh()` method to `OpponentPool`
**File**: `rl/opponent_pool.py`
**Issue**: Subprocess environments called `opponent_pool.refresh()` which didn't exist
**Fix**: Added method that reloads `pool_state.json` from disk

### 2. Fixed `None` training policy in subprocesses
**File**: `rl/multi_policy_env.py` - `_assign_policies()`
**Issue**: In SubprocVecEnv, `self._training_policy` is `None`, causing crash when assigned to training slot
**Fix**: Check if `_training_policy` is None, and if so, load from `self_play_checkpoint_path` instead

### 3. Replaced `deepcopy` with save/load
**File**: `rl/multi_policy_env.py` - `_create_frozen_copy()`
**Issue**: `deepcopy` on PyTorch models in multiprocessing can cause crashes
**Fix**: Use temporary file save/load to create independent copy

### 4. Added warning for None policies
**File**: `rl/multi_policy_env.py` - `PolicySlot.get_action()`
**Issue**: Silent failures when policy is None
**Fix**: Added warning to help debug

## Potential Remaining Issues

### Issue 1: Empty Pool + No Self-Play Checkpoint
**Symptom**: Subprocess has no policy to load
**When**: Pool is empty AND `self_play_checkpoint_path` is None
**Result**: All policies will be None, using random actions

**Solution**: Ensure one of these is true:
- Pool has at least one checkpoint before starting SubprocVecEnv
- `self_play_checkpoint_path` is provided and exists
- OR accept random opponents initially

### Issue 2: Checkpoint Path Doesn't Exist Yet
**Symptom**: `_get_self_play_policy()` returns None because checkpoint file doesn't exist
**When**: First episode before `OpponentPoolCallback` has saved the checkpoint
**Result**: Training slot will have None policy

**Solution**: Pre-create the self-play checkpoint:
```python
if args.multi_policy and args.self_play_prob > 0 and use_subproc:
    # Save initial checkpoint before starting training
    if not os.path.exists(self_play_checkpoint_path + ".zip"):
        print(f"Creating initial self-play checkpoint at {self_play_checkpoint_path}")
        model.save(self_play_checkpoint_path)
```

### Issue 3: Pickling Issues
**Symptom**: EOFError during subprocess creation
**Possible causes**:
- Lambda functions in environment factory
- Unpicklable objects captured in closure
- Large objects being pickled

**Current mitigation**: Environment factory only captures simple values (strings, ints, floats)

## Diagnostic Steps

### Step 1: Run the test script
```bash
python scripts/test_subprocess.py
```

This will test:
1. Basic SubprocVecEnv (no multi-policy) - should work
2. SubprocVecEnv with MultiPolicyBusEnv - identifies where crash occurs

### Step 2: Check the output
- If Test 1 fails: Issue with basic environment or SubprocVecEnv setup
- If Test 2 fails: Issue specific to MultiPolicyBusEnv in subprocess

### Step 3: Add more debugging
If the test script doesn't reveal the issue, add print statements in `_init()`:
```python
def _init():
    print("[DEBUG] Step 1: Creating BusEnv")
    env = BusEnv(num_players=_num_players)
    
    print("[DEBUG] Step 2: Creating OpponentPool")
    subprocess_pool = OpponentPool(...)
    
    print("[DEBUG] Step 3: Creating MultiPolicyBusEnv")
    env = MultiPolicyBusEnv(...)
    
    print("[DEBUG] Step 4: Returning environment")
    return env
```

The last print before the crash will show where the issue is.

## Recommended Training Command

For initial testing with SubprocVecEnv:

```bash
# Option 1: Use DummyVecEnv (single process, easier to debug)
python scripts/train.py \
    --multi-policy \
    --n-envs 1 \
    --total-timesteps 10000 \
    --self-play-prob 0.0

# Option 2: SubprocVecEnv with pre-seeded pool
# First, create and seed the pool manually, then:
python scripts/train.py \
    --multi-policy \
    --n-envs 4 \
    --load-pool-dir logs/previous_run/opponent_pool \
    --total-timesteps 10000 \
    --self-play-prob 0.0
```

## Summary of Root Causes

The EOFError in SubprocVecEnv is typically caused by:

1. **Exception during environment creation** - subprocess crashes before it can communicate
2. **Unpicklable objects** - objects that can't be serialized for multiprocessing
3. **Missing dependencies** - subprocess can't import required modules

Our fixes address:
- ✓ Missing `refresh()` method
- ✓ None training policy in subprocess
- ✓ deepcopy issues with PyTorch models
- ⚠️ Empty pool + no checkpoint (will use random actions with warning)

If the error persists, run the diagnostic script and check which step fails.
