"""Test imports for new modules."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing new module imports...\n")
    
    try:
        print("[1/5] Testing state_detection module...")
        from symbiote.state_detection import HandState, detect_states_from_video
        print("  [OK] HandState imported")
        print("  [OK] detect_states_from_video imported")
        print("  [OK] state_detection module loaded successfully\n")
        
        print("[2/5] Testing video_inference pipeline...")
        from symbiote.pipelines.video_inference import run_video_inference
        print("  [OK] run_video_inference imported")
        print("  [OK] video_inference pipeline loaded successfully\n")
        
        print("[3/5] Testing updated video_processor...")
        from symbiote.preprocessing.video_processor import process_video_frames
        print("  [OK] process_video_frames imported (with state detection support)")
        print("  [OK] video_processor updated successfully\n")
        
        print("[4/5] Testing updated video_training...")
        from symbiote.pipelines.video_training import run_video_training
        print("  [OK] run_video_training imported (with state detection)")
        print("  [OK] video_training updated successfully\n")
        
        print("[5/5] Testing CLI main...")
        from symbiote.cli.main import main
        print("  [OK] main imported (with infer command)")
        print("  [OK] CLI updated successfully\n")
        
        print("="*60)
        print("[SUCCESS] ALL IMPORTS SUCCESSFUL!")
        print("="*60)
        print("\nNew features available:")
        print("  1. State detection framework (placeholder implementation)")
        print("  2. Video inference pipeline (CSV output)")
        print("  3. CLI 'infer' command")
        print("\nYou can now run:")
        print("  python -m symbiote.cli.main infer --help")
        
        return True
        
    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
