import sys
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import cellpose
from cellpose import models

def run_diagnostic():
    print("="*50)
    print("🔬 DIAGNÓSTICO SIMPLIFICADO")
    print("="*50)
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    libs = [torch, torchvision, cv2, np, pd, cellpose]
    for lib in libs:
        try:
            print(f"✅ {lib.__name__} OK")
        except:
            print(f"❌ Problema com {lib}")

    print("\n🎯 TESTE DE INFERÊNCIA (CPU):")
    try:
        model = models.CellposeModel(gpu=False, model_type='nuclei')
        test_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        masks, _, _ = model.eval(test_img, diameter=30, channels=[0,0])
        print(f"✅ Sucesso! Objetos detectados no teste: {masks.max()}")
    except Exception as e:
        print(f"❌ Erro no teste funcional: {e}")

    print("="*50)

if __name__ == "__main__":
    run_diagnostic()
