# app.py - Streamlit App para Análise de Partículas de Nióbio - VERSÃO OTIMIZADA

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose import models
from skimage.measure import regionprops, regionprops_table, find_contours
from skimage.color import label2rgb
from PIL import Image
import base64
from io import BytesIO
import warnings
import logging
warnings.filterwarnings('ignore')

# Configuração de Logs
logging.basicConfig(level=logging.INFO)

# Configuração da página
st.set_page_config(
    page_title="Análise de Partículas de Nióbio - SEM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título da aplicação
st.title("🔬 Análise de Partículas de Nióbio - SEM")
st.markdown("""
Esta aplicação utiliza Cellpose para segmentação e análise de partículas de nióbio em imagens SEM.
Versão otimizada com paridade Kaggle e diagnóstico robusto.
""")

# Sidebar - Configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Modo de Teste
    test_mode = st.checkbox("🔧 Modo Teste (imagem sintética)", help="Usa uma imagem gerada para testar o pipeline")
    
    # Upload da imagem
    uploaded_file = st.file_uploader("📤 Carregue uma imagem SEM", 
                                     type=['tif', 'tiff', 'png', 'jpg', 'jpeg']) if not test_mode else None
    
    st.markdown("---")
    st.subheader("🎯 Presets de Nióbio")
    preset = st.selectbox(
        "Escolha um preset",
        ["Nuclei_Conservador", "Sensível", "Personalizado"],
        help="Nuclei_Conservador: Recomendado para a maioria das detecções"
    )

    st.subheader("🔧 Parâmetros do Modelo")
    
    # Lógica de presets
    if preset == "Nuclei_Conservador":
        def_model, def_flow, def_prob, def_diam, def_size = "nuclei", 0.6, 0.2, 60, 20
    elif preset == "Sensível":
        def_model, def_flow, def_prob, def_diam, def_size = "cyto2", 0.4, -0.5, 80, 20
    else:
        def_model, def_flow, def_prob, def_diam, def_size = "nuclei", 0.6, 0.2, 60, 20

    model_type = st.selectbox("Tipo de Modelo", ["nuclei", "cyto", "cyto2"], 
                              index=["nuclei", "cyto", "cyto2"].index(def_model))
    
    col1, col2 = st.columns(2)
    with col1:
        flow_threshold = st.slider("Flow Threshold", 0.0, 1.0, def_flow, 0.1)
    with col2:
        cellprob_threshold = st.slider("Cellprob Threshold", -6.0, 6.0, def_prob, 0.1)
    
    diameter = st.number_input("Diâmetro esperado (px)", 10, 200, def_diam, 10)
    min_size = st.number_input("Tamanho mínimo (px)", 1, 100, def_size, 5)
    use_gpu = st.checkbox("Utilizar GPU (CUDA)", value=False)
    
    st.markdown("---")
    st.subheader("📏 Filtros Físicos")
    
    microns_per_pixel = st.number_input("µm por pixel", 0.001, 1.0, 0.02, 0.001, format="%.3f")
    
    col3, col4 = st.columns(2)
    with col3:
        min_area_um2 = st.number_input("Área mín (µm²)", 0.001, 10.0, 0.01, 0.01, format="%.3f")
    with col4:
        max_area_um2 = st.number_input("Área máx (µm²)", 0.1, 100.0, 15.0, 0.1, format="%.1f")
    
    min_circularity = st.slider("Circularidade mín", 0.0, 1.0, 0.15, 0.05)
    max_aspect_ratio = st.slider("Aspect Ratio máx", 1.0, 10.0, 4.0, 0.5)
    
    st.markdown("---")
    process_button = st.button("🚀 Processar Imagem", type="primary", use_container_width=True)

# --- FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO ---

def robust_load_image(uploaded_file, test_mode=False):
    """Carregamento de imagem com múltiplos fallbacks e tratamento de profundidade"""
    if test_mode:
        test_img = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(test_img, (150, 150), 30, 200, -1)
        cv2.circle(test_img, (350, 200), 40, 220, -1)
        cv2.circle(test_img, (250, 350), 25, 180, -1)
        return test_img

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # 1. Tentar OpenCV com flags de alta profundidade
    img = cv2.imdecode(file_bytes, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    
    # 2. Fallback para PIL se OpenCV falhar
    if img is None:
        try:
            pil_img = Image.open(BytesIO(file_bytes))
            img = np.array(pil_img)
        except Exception as e:
            st.error(f"Erro ao carregar imagem: {e}")
            return None

    # Redimensionar se for muito grande para evitar estouro de memória
    MAX_SIZE = 1536
    if img.shape[0] > MAX_SIZE or img.shape[1] > MAX_SIZE:
        scale = MAX_SIZE / max(img.shape[0], img.shape[1])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        st.warning(f"Imagem redimensionada para {new_size[0]}x{new_size[1]} para performance.")

    return img

def preprocess_image(img):
    """Pré-processamento idêntico ao Kaggle com conversão robusta"""
    # Converter para escala de cinza
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = img[:,:,0]
    
    # Normalização robusta (0-1)
    img_float = img.astype(np.float32)
    img_min = img_float.min()
    img_max = img_float.max()
    img_norm = (img_float - img_min) / (img_max - img_min + 1e-8)
    
    # Inversão para SEM (fundo claro -> partículas claras)
    if np.mean(img_norm) > 0.5:
        img_norm = 1 - img_norm
    
    # Contraste
    img_norm = np.clip(img_norm * 1.3, 0, 1)
    
    # Imagem original normalizada a uint8 para display
    img_display = ((img_float - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
    
    return img_norm, img_display

def get_model(use_gpu, model_type):
    """Carregamento de modelo com fallback automático para CPU"""
    try:
        if use_gpu:
            import torch
            if torch.cuda.is_available():
                model = models.CellposeModel(gpu=True, model_type=model_type)
                return model, "GPU"
            else:
                st.warning("GPU selecionada mas não disponível. Usando CPU.")
        
        model = models.CellposeModel(gpu=False, model_type=model_type)
        return model, "CPU"
    except Exception as e:
        st.error(f"Erro ao carregar modelo Cellpose: {e}")
        return None, None

def analyze_particles_v2(masks, params):
    """Análise com métricas idênticas ao Kaggle"""
    props = regionprops_table(masks, properties=[
        'label', 'area', 'perimeter', 'major_axis_length',
        'minor_axis_length', 'centroid', 'orientation',
        'eccentricity', 'bbox', 'equivalent_diameter_area'
    ])
    df = pd.DataFrame(props)
    
    if df.empty: return df, df, np.zeros_like(masks)

    # Métricas avançadas
    df['area_um2'] = df['area'] * (params['microns_per_pixel'] ** 2)
    df['perimeter_um'] = df['perimeter'] * params['microns_per_pixel']
    df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-8)
    df['aspect_ratio'] = df['major_axis_length'] / (df['minor_axis_length'] + 1e-8)
    df['diameter_um'] = df['equivalent_diameter_area'] * params['microns_per_pixel']
    
    # Filtros
    df_filtered = df[
        (df['area_um2'] > params['min_area_um2']) &
        (df['area_um2'] < params['max_area_um2']) &
        (df['circularity'] > params['min_circularity']) &
        (df['aspect_ratio'] < params['max_aspect_ratio']) &
        (df['area'] > 5)
    ].copy().reset_index(drop=True)
    
    # Máscara filtrada
    filtered_masks = np.zeros_like(masks)
    for idx, lbl in enumerate(df_filtered['label']):
        filtered_masks[masks == lbl] = idx + 1
        
    return df, df_filtered, filtered_masks

# --- EXECUÇÃO PRINCIPAL ---

if (uploaded_file is not None or test_mode) and process_button:
    try:
        # 1. Carregamento
        img = robust_load_image(uploaded_file, test_mode)
        if img is None: st.stop()
        
        # 2. Debug Informação Inicial
        with st.expander("🔍 Debug - Informações da Imagem"):
            st.write(f"Dtype: {img.dtype}, Shape: {img.shape}")
            st.write(f"Valores: min={img.min()}, max={img.max()}, mean={img.mean():.2f}")
            st.image(img, caption="Imagem Carregada (Original)", use_column_width=True, clamp=True)

        # 3. Processamento
        img_norm, img_display = preprocess_image(img)
        
        # 4. Inferência
        model, device = get_model(use_gpu, model_type)
        if model is None: st.stop()
        
        with st.spinner(f"Segmentando via {device}..."):
            masks, flows, styles = model.eval(
                img_norm, diameter=diameter, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, normalize=True,
                min_size=min_size, augment=True
            )
        
        # 5. Análise
        params = {
            'microns_per_pixel': microns_per_pixel, 'min_area_um2': min_area_um2,
            'max_area_um2': max_area_um2, 'min_circularity': min_circularity,
            'max_aspect_ratio': max_aspect_ratio
        }
        df_raw, df_filtered, filtered_masks = analyze_particles_v2(masks, params)
        
        # --- EXIBIÇÃO ---
        st.header("📊 Resultado da Análise")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Brutos", masks.max())
        col2.metric("Filtrados", len(df_filtered))
        col3.metric("Área Média", f"{df_filtered['area_um2'].mean():.3f} µm²" if not df_filtered.empty else "N/A")
        col4.metric("Hardware", device)

        # Visualizações
        tab1, tab2, tab3 = st.tabs(["🖼️ Overlay", "📈 Estatísticas", "📋 Dados"])
        
        with tab1:
            overlay_rgb = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            contours = find_contours(filtered_masks.astype(float), 0.5)
            # Simplificando overlay para performance
            for c in contours:
                c = np.flip(c, axis=1).astype(np.int32)
                cv2.polylines(overlay_rgb, [c], True, (0, 255, 0), 2)
            
            c1, c2 = st.columns(2)
            c1.image(img_display, caption="Original Normalizada", use_column_width=True)
            c2.image(overlay_rgb, caption="Detecções Filtradas", use_column_width=True)

        with tab2:
            if not df_filtered.empty:
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].hist(df_filtered['area_um2'], bins=20, color='skyblue', edgecolor='black')
                ax[0].set_title("Distribuição de Áreas (µm²)")
                ax[1].scatter(df_filtered['area_um2'], df_filtered['circularity'], alpha=0.5)
                ax[1].set_title("Circularidade vs Área")
                st.pyplot(fig)
            else:
                st.warning("Sem dados suficientes para gráficos.")

        with tab3:
            st.dataframe(df_filtered.head(100), use_container_width=True)
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar CSV", csv, "analise_mev.csv", "text/csv")

    except Exception as e:
        st.error(f"Erro crítico no processamento: {e}")
        logging.exception("Erro no app")
else:
    st.info("👈 Ajuste as configurações e clique em 'Processar Imagem' para iniciar.")
    st.markdown("""
    ### 📝 Como usar:
    1. Carregue sua imagem SEM (TIF, PNG, JPG).
    2. Escolha o preset adequado (ou ajuste manualmente).
    3. Defina a escala (µm por pixel).
    4. Clique em Processar.
    
    *Dica: O Modo Teste gera uma imagem artificial para validar se o modelo está funcionando no servidor.*
    """)