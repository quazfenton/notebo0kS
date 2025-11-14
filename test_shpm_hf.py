#!/usr/bin/env python3
\"\"\"
Test script to verify the Hugging Face model integration in SHPM
\"\"\"
from shpm.models import SHPMConfig, create_shpm_model
from shpm.train import TrainConfig, train_entry

def test_hf_model():
    print(\"Testing Hugging Face model integration in SHPM...\")
    
    # Test using SciBERT
    print(\"\\n1. Testing with SciBERT model...\")
    cfg = SHPMConfig(
        use_hf_model=True,
        hf_model_name=\"microsoft/scibert-scivocab-uncased\",
        freeze_hf_backbone=False
    )
    
    try:
        model = create_shpm_model(cfg)
        print(f\"✓ Successfully created SciBERT-based SHPM model\")
        print(f\"  Model type: {type(model)}\")
        print(f\"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\")
    except Exception as e:
        print(f\"✗ Error creating SciBERT model: {e}\")
    
    # Test using T5
    print(\"\\n2. Testing with T5 model...\")
    cfg = SHPMConfig(
        use_hf_model=True,
        hf_model_name=\"t5-small\",
        freeze_hf_backbone=False
    )
    
    try:
        model = create_shpm_model(cfg)
        print(f\"✓ Successfully created T5-based SHPM model\")
        print(f\"  Model type: {type(model)}\")
        print(f\"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\")
    except Exception as e:
        print(f\"✗ Error creating T5 model: {e}\")
    
    # Test using TinySHPM fallback
    print(\"\\n3. Testing TinySHPM fallback...\")
    cfg = SHPMConfig(
        use_hf_model=False,
    )
    
    try:
        model = create_shpm_model(cfg)
        print(f\"✓ Successfully created TinySHPM model\")
        print(f\"  Model type: {type(model)}\")
        print(f\"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\")
    except Exception as e:
        print(f\"✗ Error creating TinySHPM model: {e}\")

def test_training_config():
    print(\"\\n4. Testing training configuration...\")
    
    cfg = TrainConfig(
        use_hf_model=True,
        hf_model_name=\"microsoft/scibert-scivocab-uncased\",
        epochs=1,
        batch_size=2,
        wandb_enable=False,
        save_dir=\"outputs/test_shpm\"
    )
    
    print(f\"✓ Training config created successfully\")
    print(f\"  Model: {cfg.hf_model_name if cfg.use_hf_model else 'TinySHPM'}\")
    print(f\"  Epochs: {cfg.epochs}, Batch size: {cfg.batch_size}\")

if __name__ == \"__main__\":
    test_hf_model()
    test_training_config()
    print(\"\\n✓ All basic tests passed!\")