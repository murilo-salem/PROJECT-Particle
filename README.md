# Pipeline de Segmentação e Análise de Partículas em Imagens SEM/MEV Router-aware com Cellpose

Este projeto implementa **um pipeline completo de segmentação, análise morfológica e visualização de partículas em imagens de Microscopia Eletrônica de Varredura (SEM/MEV)** utilizando **deep learning com Cellpose (v4.2+)**.  

O foco principal é **detecção automática e caracterização quantitativa de partículas**, com aplicação direta em ciência dos materiais, especialmente pós metálicos (ex.: nióbio), nanopartículas e superfícies complexas.

---

## 🎯 Objetivo Principal

- Segmentar partículas em imagens SEM usando **Cellpose**  
- Extrair métricas morfológicas e físicas relevantes  
- Filtrar partículas com critérios reprodutíveis  
- Gerar visualizações, gráficos e relatórios estruturados  

---

## 🧠 Visão Geral do Pipeline

> O pipeline foi desenvolvido inicialmente em **notebook**, otimizado para execução em **Kaggle (GPU)**, mas facilmente adaptável para ambientes locais.

### 1️⃣ Instalação e Importação
- Instalação das dependências principais:
  - `cellpose`
  - `opencv-python`
  - `scikit-image`
  - `numpy`, `pandas`
  - `matplotlib`
- Importação modular para processamento, análise e visualização.

---

### 2️⃣ Configuração do Usuário
Parâmetros facilmente ajustáveis:
- Caminho da imagem SEM  
- Calibração espacial (**µm por pixel**)  
- Prefixo de saída  
- Filtros de área mínima e máxima  

---

### 3️⃣ Pré-processamento da Imagem
Tratamento específico para imagens SEM:
- Leitura em **escala de cinza (16 bits)**  
- Normalização de intensidade  
- Inversão automática (fundo escuro / partículas claras)  
- Ajuste de contraste para melhorar a segmentação  

<!-- 
📷 **Placeholder – Imagem original vs. pré-processada**
-->

---

### 4️⃣ Segmentação com Cellpose
- Uso do modelo **cyto3** (GPU habilitada)  
- Inferência automática para geração das máscaras  
- Estimativa automática do diâmetro médio das partículas  

<!-- 
📷 **Placeholder – Máscaras geradas pelo Cellpose**
-->

---

### 5️⃣ Visualização Intermediária
- Imagem original  
- Imagem pré-processada  
- Máscaras brutas  
- Overlay básico para validação rápida  

---

### 6️⃣ Pós-processamento e Análise
Extração de propriedades usando `regionprops`:
- Área  
- Perímetro  
- Circularidade  
- Excentricidade  
- Razão de aspecto  
- Diâmetro equivalente  

Conversão automática para **unidades físicas (µm², µm)**.

Aplicação de filtros:
- Área mínima / máxima  
- Circularidade  
- Outros critérios geométricos  

---

### 7️⃣ Visualização Detalhada
- Overlays coloridos com:
  - Contornos  
  - Eixo maior  
  - Centróides  
- Histogramas de distribuição de tamanho  
- Gráfico **Circularidade × Área**

<!--
📷 **Placeholder – Overlay final com métricas**


📊 **Placeholder – Gráficos estatísticos**
-->

---

### 8️⃣ Estatísticas e Relatórios
- Estatísticas resumidas:
  - Média, desvio padrão, área total  
- Agrupamento por faixas de tamanho  
- Exportação de tabelas estruturadas  

---

### 9️⃣ Salvamento de Resultados
- Imagens finais com overlays  
- Máscaras brutas e filtradas (`.npy`)  
- CSVs contendo:
  - Métricas das partículas  
  - Estatísticas resumidas  
  - Configurações do experimento  

---

## 🛠️ Técnicas e Ferramentas Utilizadas

- **Cellpose** – Segmentação baseada em deep learning  
- **OpenCV & scikit-image** – Processamento e análise de imagens  
- **Análise Morfológica** – Circularidade, excentricidade, razão de aspecto  
- **Matplotlib** – Visualização científica  
- **CSV / NumPy** – Exportação estruturada de dados  

---

## 🧪 Contexto de Aplicação

Este pipeline é ideal para:
- Ciência e engenharia de materiais  
- Caracterização de pós metálicos e nanopartículas  
- Geologia, biologia e superfícies complexas  
- Automação e reprodutibilidade em análise SEM  

---

## ⚠️ Observações Importantes
- Otimizado para **GPU (Kaggle)**  
- Inclui tratamento específico para imagens SEM  
- Totalmente flexível para ajustes de parâmetros  

---

# 🌐 Aplicação Web Interativa (Streamlit)

Além do pipeline em notebook, o projeto inclui uma **aplicação web interativa**, desenvolvida com **Streamlit**, que torna toda a análise acessível para usuários finais, sem necessidade de editar código.

---

## 🎯 Objetivo da Aplicação
- Interface gráfica simples para análise de imagens SEM  
- Execução interativa da segmentação com Cellpose  
- Exploração visual e tabular dos resultados  
- Exportação fácil dos dados e imagens  

---

## 🧩 Funcionalidades Principais

### 📤 Entrada de Dados
- Upload de imagens SEM (`.tif`, `.png`, `.jpg`)  
- Modo de teste com imagem sintética  

### 🎛️ Controles Interativos
- Ajuste do diâmetro esperado  
- Thresholds do Cellpose  
- Escala física (µm/pixel)  
- Filtros geométricos (área, circularidade, aspecto)  

### 👁️ Visualização
- Overlay dinâmico das partículas detectadas  
- Gráficos interativos de distribuição  
- Tabela com métricas detalhadas  

<!--
📷 **Placeholder – Interface do app**
-->

---

### 📥 Exportação
- Download de:
  - CSV com métricas  
  - Imagem final com overlays  

---

## 📂 Estrutura do Projeto

```text
├── app.py              # Aplicação Streamlit
├── code.py             # Versão otimizada do app
├── test_cellpose.py    # Diagnóstico de instalação
├── requirements.txt    # Dependências do projeto
├── pip_out.txt         # Log de instalação
├── pip_err.txt         # Log de erros
└── README.md
