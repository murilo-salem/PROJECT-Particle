# app.py - Streamlit App para Análise de Partículas de Nióbio

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose import models
from skimage.measure import regionprops, regionprops_table, find_contours
from skimage.color import label2rgb
import tempfile
import os
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

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
Carregue uma imagem, ajuste os parâmetros e visualize os resultados.
""")

# Sidebar - Configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Upload da imagem
    uploaded_file = st.file_uploader("📤 Carregue uma imagem SEM", 
                                     type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])
    
    st.markdown("---")
    st.subheader("🔧 Parâmetros do Modelo")
    
    # Parâmetros do Cellpose
    model_type = st.selectbox(
        "Tipo de Modelo",
        ["nuclei", "cyto", "cyto2"],
        index=0,
        help="nuclei: otimizado para núcleos, cyto: citoplasma, cyto2: versão melhorada"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        flow_threshold = st.slider(
            "Flow Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Limiar para o fluxo (mais baixo = mais sensível)"
        )
    
    with col2:
        cellprob_threshold = st.slider(
            "Cellprob Threshold",
            min_value=-6.0,
            max_value=6.0,
            value=0.2,
            step=0.1,
            help="Limiar de probabilidade (negativo = mais sensível)"
        )
    
    diameter = st.number_input(
        "Diâmetro esperado (pixels)",
        min_value=10,
        max_value=200,
        value=60,
        step=10,
        help="Diâmetro médio esperado das partículas"
    )
    
    min_size = st.number_input(
        "Tamanho mínimo (pixels)",
        min_value=1,
        max_value=100,
        value=20,
        step=5,
        help="Tamanho mínimo para considerar como partícula"
    )
    
    st.markdown("---")
    st.subheader("📏 Filtros Físicos")
    
    microns_per_pixel = st.number_input(
        "Micrômetros por pixel",
        min_value=0.001,
        max_value=1.0,
        value=0.02,
        step=0.001,
        format="%.3f",
        help="Calibração da imagem SEM"
    )
    
    col3, col4 = st.columns(2)
    with col3:
        min_area_um2 = st.number_input(
            "Área mínima (µm²)",
            min_value=0.001,
            max_value=10.0,
            value=0.01,
            step=0.01,
            format="%.3f"
        )
    
    with col4:
        max_area_um2 = st.number_input(
            "Área máxima (µm²)",
            min_value=0.1,
            max_value=100.0,
            value=15.0,
            step=0.1,
            format="%.1f"
        )
    
    min_circularity = st.slider(
        "Circularidade mínima",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help="0 = formas muito irregulares, 1 = círculo perfeito"
    )
    
    max_aspect_ratio = st.slider(
        "Razão de aspecto máxima",
        min_value=1.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
        help="1 = perfeitamente redondo, valores maiores = mais alongado"
    )
    
    st.markdown("---")
    
    # Botão de processamento
    process_button = st.button("🚀 Processar Imagem", type="primary", use_container_width=True)

# Função para processar a imagem
def process_image(img, params):
    """Processa a imagem com os parâmetros fornecidos"""
    
    # Converter para escala de cinza se necessário
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = img[:,:,0]
    
    # Normalização
    img_float = img.astype(np.float32)
    img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
    
    # Inverter se necessário
    if np.mean(img_norm) > 0.5:
        img_norm = 1 - img_norm
    
    # Melhorar contraste
    img_norm = np.clip(img_norm * 1.3, 0, 1)
    
    return img_norm, img

# Função para segmentação com Cellpose
def segment_image(img_norm, params):
    """Segmenta a imagem usando Cellpose"""
    
    # Carregar modelo
    model = models.CellposeModel(gpu=False, model_type=params['model_type'])
    
    # Executar segmentação
    masks, flows, styles = model.eval(
        img_norm,
        diameter=params['diameter'],
        flow_threshold=params['flow_threshold'],
        cellprob_threshold=params['cellprob_threshold'],
        normalize=True,
        min_size=params['min_size'],
        augment=True
    )
    
    return masks

# Função para análise das partículas
def analyze_particles(masks, img_norm, params):
    """Analisa as partículas detectadas"""
    
    # Calcular propriedades
    props = regionprops_table(
        masks,
        properties=[
            'label', 'area', 'perimeter', 'major_axis_length',
            'minor_axis_length', 'centroid', 'orientation',
            'eccentricity', 'bbox', 'equivalent_diameter_area'
        ]
    )
    
    df = pd.DataFrame(props)
    
    # Calcular métricas avançadas
    df['area_um2'] = df['area'] * (params['microns_per_pixel'] ** 2)
    df['perimeter_um'] = df['perimeter'] * params['microns_per_pixel']
    df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-8)
    df['aspect_ratio'] = df['major_axis_length'] / (df['minor_axis_length'] + 1e-8)
    df['diameter_um'] = df['equivalent_diameter_area'] * params['microns_per_pixel']
    
    # Aplicar filtros
    df_filtered = df[
        (df['area_um2'] > params['min_area_um2']) &
        (df['area_um2'] < params['max_area_um2']) &
        (df['circularity'] > params['min_circularity']) &
        (df['aspect_ratio'] < params['max_aspect_ratio']) &
        (df['area'] > 5)
    ].copy().reset_index(drop=True)
    
    # Criar máscara filtrada
    filtered_masks = np.zeros_like(masks)
    new_id = 1
    
    for idx, lbl in enumerate(df_filtered['label']):
        filtered_masks[masks == lbl] = new_id
        new_id += 1
    
    return df, df_filtered, filtered_masks

# Função para criar overlay
def create_overlay(img_norm, masks, df_filtered):
    """Cria overlay com contornos das partículas"""
    
    overlay_rgb = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Paleta de cores
    colors = [
        (255, 100, 100),  # Vermelho claro
        (100, 255, 100),  # Verde claro
        (100, 100, 255),  # Azul claro
        (255, 255, 100),  # Amarelo
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Ciano
        (255, 200, 100),  # Laranja
        (200, 100, 255),  # Roxo
    ]
    
    for idx, (_, region) in enumerate(df_filtered.iterrows()):
        label = int(region['label'])
        mask_r = masks == label
        
        # Escolher cor
        color = colors[idx % len(colors)]
        
        # Contorno
        contours = find_contours(mask_r.astype(float), 0.5)
        for c in contours:
            if len(c) > 2:
                c = np.flip(c, axis=1).astype(np.int32)
                cv2.polylines(overlay_rgb, [c], True, color, 2)
                cv2.polylines(overlay_rgb, [c], True, (255, 255, 255), 1)
        
        # Centroide
        y0, x0 = region['centroid-0'], region['centroid-1']
        cv2.circle(overlay_rgb, (int(x0), int(y0)), 4, (255, 255, 255), -1)
        cv2.circle(overlay_rgb, (int(x0), int(y0)), 2, color, -1)
    
    return overlay_rgb

# Função para criar gráficos
def create_plots(df_filtered, img_shape, microns_per_pixel):
    """Cria os gráficos de análise"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if len(df_filtered) > 0:
        # 1. Histograma de tamanhos
        axes[0, 0].hist(df_filtered['area_um2'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(df_filtered['area_um2'].mean(), color='red', linestyle='--', 
                          label=f'Média: {df_filtered["area_um2"].mean():.2f} µm²')
        axes[0, 0].set_xlabel('Área (µm²)')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].set_title('Distribuição de Tamanhos')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter: Área vs Circularidade
        scatter = axes[0, 1].scatter(df_filtered['area_um2'], df_filtered['circularity'], 
                                    c=df_filtered['aspect_ratio'], cmap='viridis', 
                                    alpha=0.7, s=50, edgecolors='black')
        axes[0, 1].set_xlabel('Área (µm²)')
        axes[0, 1].set_ylabel('Circularidade')
        axes[0, 1].set_title('Circularidade vs Área')
        plt.colorbar(scatter, ax=axes[0, 1]).set_label('Razão de Aspecto')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Gráfico de pizza por faixa de tamanho
        bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        labels = ['<0.5', '0.5-1.0', '1.0-2.0', '2.0-5.0', '5.0-10.0', '>10.0']
        df_filtered['size_bin'] = pd.cut(df_filtered['area_um2'], bins=bins, labels=labels)
        size_dist = df_filtered['size_bin'].value_counts().sort_index()
        
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(size_dist)))
        wedges, texts, autotexts = axes[0, 2].pie(size_dist.values, labels=size_dist.index, 
                                                 autopct='%1.1f%%', colors=colors_pie,
                                                 startangle=90, counterclock=False)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        axes[0, 2].set_title('Distribuição por Tamanho')
        
        # 4. Boxplot de métricas
        metrics_to_plot = ['area_um2', 'circularity', 'aspect_ratio']
        data_to_plot = [df_filtered[metric] for metric in metrics_to_plot]
        labels_box = ['Área (µm²)', 'Circularidade', 'Razão Aspecto']
        
        bp = axes[1, 0].boxplot(data_to_plot, labels=labels_box, patch_artist=True)
        colors_box = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
        axes[1, 0].set_title('Distribuição das Métricas')
        axes[1, 0].set_ylabel('Valor')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. Scatter: Diâmetro vs Razão de Aspecto
        axes[1, 1].scatter(df_filtered['diameter_um'], df_filtered['aspect_ratio'], 
                          c=df_filtered['circularity'], cmap='plasma', 
                          alpha=0.7, s=50, edgecolors='black')
        axes[1, 1].set_xlabel('Diâmetro (µm)')
        axes[1, 1].set_ylabel('Razão de Aspecto')
        axes[1, 1].set_title('Forma vs Tamanho')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Estatísticas resumidas
        axes[1, 2].axis('off')
        stats_text = f"""
        📊 RESUMO ESTATÍSTICO
        
        Total de Partículas: {len(df_filtered)}
        
        📏 TAMANHO:
        • Mínima: {df_filtered['area_um2'].min():.2f} µm²
        • Máxima: {df_filtered['area_um2'].max():.2f} µm²
        • Média: {df_filtered['area_um2'].mean():.2f} µm²
        • Mediana: {df_filtered['area_um2'].median():.2f} µm²
        
        🔵 FORMA:
        • Circularidade média: {df_filtered['circularity'].mean():.3f}
        • Razão aspecto: {df_filtered['aspect_ratio'].mean():.2f}:1
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
                       verticalalignment='center', transform=axes[1, 2].transAxes)
    
    else:
        # Se não houver dados
        for i in range(2):
            for j in range(3):
                axes[i, j].text(0.5, 0.5, 'Sem dados para gráfico', 
                               ha='center', va='center')
                axes[i, j].set_title('Gráfico Indisponível')
    
    plt.tight_layout()
    return fig

# Função para download de dados
def get_table_download_link(df, filename):
    """Gera link para download do DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

# Função para converter imagem para base64
def get_image_download_link(img, filename):
    """Gera link para download da imagem"""
    _, buffer = cv2.imencode('.png', img)
    b64 = base64.b64encode(buffer).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download PNG</a>'
    return href

# Processamento principal
if uploaded_file is not None and process_button:
    # Ler imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    if img is not None:
        # Parâmetros
        params = {
            'model_type': model_type,
            'flow_threshold': flow_threshold,
            'cellprob_threshold': cellprob_threshold,
            'diameter': diameter,
            'min_size': min_size,
            'microns_per_pixel': microns_per_pixel,
            'min_area_um2': min_area_um2,
            'max_area_um2': max_area_um2,
            'min_circularity': min_circularity,
            'max_aspect_ratio': max_aspect_ratio
        }
        
        # Barra de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Pré-processamento
        status_text.text("🔧 Pré-processando imagem...")
        img_norm, img_original = process_image(img, params)
        progress_bar.progress(25)
        
        # Segmentação
        status_text.text("🎯 Segmentando com Cellpose...")
        masks = segment_image(img_norm, params)
        progress_bar.progress(50)
        
        # Análise
        status_text.text("📊 Analisando partículas...")
        df, df_filtered, filtered_masks = analyze_particles(masks, img_norm, params)
        progress_bar.progress(75)
        
        # Visualização
        status_text.text("🎨 Gerando visualizações...")
        overlay_rgb = create_overlay(img_norm, masks, df_filtered)
        fig = create_plots(df_filtered, img_original.shape, microns_per_pixel)
        progress_bar.progress(100)
        status_text.text("✅ Processamento concluído!")
        
        # Resultados
        st.header("📊 Resultados da Análise")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Detectado", f"{masks.max()}")
        with col2:
            st.metric("Após Filtros", f"{len(df_filtered)}")
        with col3:
            if len(df_filtered) > 0:
                st.metric("Área Média", f"{df_filtered['area_um2'].mean():.2f} µm²")
            else:
                st.metric("Área Média", "N/A")
        with col4:
            if len(df_filtered) > 0:
                st.metric("Circularidade", f"{df_filtered['circularity'].mean():.3f}")
            else:
                st.metric("Circularidade", "N/A")
        
        # Visualizações
        st.subheader("🖼️ Visualizações")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_original, caption="Imagem Original", use_column_width=True)
        
        with col2:
            st.image(overlay_rgb, caption="Partículas Detectadas", use_column_width=True)
        
        # Gráficos
        st.subheader("📈 Análise Estatística")
        st.pyplot(fig)
        
        # Dados detalhados
        st.subheader("📋 Dados Detalhados")
        
        if len(df_filtered) > 0:
            # Selecionar colunas para exibição
            display_columns = ['area_um2', 'diameter_um', 'circularity', 'aspect_ratio', 
                              'perimeter_um', 'eccentricity']
            st.dataframe(df_filtered[display_columns].head(20))
            
            # Downloads
            st.subheader("💾 Download de Resultados")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(get_table_download_link(df_filtered, "particulas_detalhadas.csv"), 
                           unsafe_allow_html=True)
            with col2:
                st.markdown(get_image_download_link(overlay_rgb, "overlay.png"), 
                           unsafe_allow_html=True)
            with col3:
                # Salvar gráfico
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="graficos.png">📥 Download Gráficos</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            # Estatísticas detalhadas
            st.subheader("📊 Estatísticas Detalhadas")
            
            if len(df_filtered) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📏 Estatísticas de Tamanho:**")
                    st.write(f"- Área total: {df_filtered['area_um2'].sum():.2f} µm²")
                    st.write(f"- Porcentagem da imagem: {df_filtered['area_um2'].sum()/(img_original.shape[0]*img_original.shape[1]*(microns_per_pixel**2))*100:.1f}%")
                    st.write(f"- Densidade: {len(df_filtered)/(img_original.shape[0]*img_original.shape[1]*(microns_per_pixel**2)):.2f} partículas/µm²")
                
                with col2:
                    st.markdown("**🔵 Uniformidade:**")
                    if df_filtered['area_um2'].mean() > 0:
                        cv_area = (df_filtered['area_um2'].std() / df_filtered['area_um2'].mean()) * 100
                        st.write(f"- CV Área: {cv_area:.1f}%")
                    if df_filtered['circularity'].mean() > 0:
                        cv_circ = (df_filtered['circularity'].std() / df_filtered['circularity'].mean()) * 100
                        st.write(f"- CV Circularidade: {cv_circ:.1f}%")
        
        else:
            st.warning("⚠️ Nenhuma partícula passou nos filtros aplicados!")
            st.info("💡 Sugestões:")
            st.write("1. Ajuste os filtros na barra lateral")
            st.write("2. Reduza os limiares de área mínima e circularidade")
            st.write("3. Verifique se a imagem possui partículas visíveis")
    
    else:
        st.error("❌ Erro ao carregar a imagem. Verifique o formato do arquivo.")
else:
    # Tela inicial
    st.info("👈 Carregue uma imagem SEM e ajuste os parâmetros na barra lateral para começar a análise.")
    
    # Exemplo de uso
    st.markdown("---")
    st.subheader("📚 Como usar:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Carregue a imagem**
        - Formatos suportados: TIF, TIFF, PNG, JPG
        - Imagens SEM em escala de cinza
        """)
    
    with col2:
        st.markdown("""
        **2. Ajuste parâmetros**
        - Modelo: nuclei para partículas pequenas
        - Flow threshold: 0.6 para conservador
        - Cellprob: 0.2 para moderado
        """)
    
    with col3:
        st.markdown("""
        **3. Configure filtros**
        - Micrômetros por pixel: calibração SEM
        - Área mín/máx: filtre por tamanho
        - Circularidade: controle a forma
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Configurações Recomendadas para Nióbio:")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("""
        **Conservador:**
        - Modelo: nuclei
        - Flow: 0.6
        - Cellprob: 0.2
        - Diâmetro: 60
        """)
    
    with rec_col2:
        st.markdown("""
        **Sensível:**
        - Modelo: cyto2
        - Flow: 0.4
        - Cellprob: -0.5
        - Diâmetro: 80
        """)

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 <b>Análise de Partículas de Nióbio - SEM</b> | Desenvolvido com Streamlit e Cellpose</p>
    <p style='font-size: 12px; color: #666;'>Versão 1.0 | Para uso em pesquisas com imagens SEM</p>
</div>
""", unsafe_allow_html=True)