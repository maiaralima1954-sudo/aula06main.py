import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuração da página do Streamlit para o público executivo
st.set_page_config(page_title="Predição de Inadimplência", layout="wide")

# ==========================================
# 1. BASE DE DADOS DE TESTE & TRATAMENTO
# ==========================================
@st.cache_data
def carregar_dados():
    # Base de dados de teste baseada no histórico anterior
    dados = {
        'ID_Cliente': ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010'],
        'Nome': ['Ana Silva', 'Bruno Santos', 'Carlos Lima', 'Daniela Oliveira', 'Eduardo Costa', 
                 'Fernanda Souza', 'Gabriel Almeida', 'Juliana Ribeiro', 'Lucas Pereira', 'Patrícia Gomes'],
        'Jan': ['Pago', 'Pago', 'Atrasado', 'Pago', 'Pago', 'Atrasado', 'Pago', 'Pago', 'Pago', 'Atrasado'],
        'Fev': ['Atrasado', 'Pago', 'Pago', 'Pago', 'Pago', 'Atrasado', 'Pago', 'Atrasado', 'Pago', 'Pago'],
        'Mar': ['Pago', 'Pago', 'Atrasado', 'Pago', 'Pago', 'Inadimplente', 'Atrasado', 'Pago', 'Pago', 'Pago'],
        'Abr': ['Atrasado', 'Pago', 'Pago', 'Atrasado', 'Pago', 'Inadimplente', 'Pago', 'Pago', 'Pago', 'Atrasado'],
        'Mai': ['Pago', 'Pago', 'Atrasado', 'Atrasado', 'Pago', 'Inadimplente', 'Pago', 'Atrasado', 'Pago', 'Pago'],
        'Jun': ['Inadimplente', 'Pago', 'Pago', 'Inadimplente', 'Atrasado', 'Inadimplente', 'Pago', 'Inadimplente', 'Pago', 'Atrasado']
    }
    df = pd.DataFrame(dados)
    
    # Mapeamento estrito para a IA: Pago=0, Atrasado=1, Inadimplente=2
    mapa_status = {'Pago': 0, 'Atrasado': 1, 'Inadimplente': 2}
    df_numerico = df.copy()
    for mes in ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun']:
        df_numerico[mes] = df_numerico[mes].map(mapa_status)
        
    return df, df_numerico

df_original, df_num = carregar_dados()

# ==========================================
# 2. INTELIGÊNCIA ARTIFICIAL (TENSORFLOW)
# ==========================================
@st.cache_resource
def treinar_modelo(df_num):
    # Features (X): Histórico de Janeiro a Maio
    X = df_num[['Jan', 'Fev', 'Mar', 'Abr', 'Mai']].values.astype(np.float32)
    # Target (y): Se virou inadimplente em Junho (1 se Inadimplente, 0 caso contrário)
    y = (df_num['Jun'] == 2).astype(np.float32).values

    # Construindo uma Rede Neural Simples para Classificação Binária
    model = Sequential([
        Dense(8, activation='relu', input_shape=(5,)),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid') # Retorna uma probabilidade entre 0% e 100%
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Treino rápido simulado para a base de teste
    model.fit(X, y, epochs=150, verbose=0)
    return model

modelo_ia = treinar_modelo(df_num)

# Predição em tempo real para todos os clientes
X_atual = df_num[['Jan', 'Fev', 'Mar', 'Abr', 'Mai']].values.astype(np.float32)
probabilidades = modelo_ia.predict(X_atual).flatten()
df_original['Probabilidade_Inadimplencia'] = probabilidades

# ==========================================
# 3. INTERFACE GRÁFICA (STREAMLIT)
# ==========================================
st.title("📊 Painel de Previsão de Inadimplência")
st.subheader("Análise Preditiva baseada em Atrasos Alternados")
st.markdown("---")

# Métricas de Topo (KPIs para Diretores)
clientes_risco_alto = df_original[df_original['Probabilidade_Inadimplencia'] > 0.50].shape[0]
impacto_estimado = clientes_risco_alto * 350.00 # Exemplo de ticket médio de R$ 350

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Risco Geral Médio", value=f"{probabilidades.mean()*100:.1f}%")
with col2:
    st.metric(label="Clientes em Alerta Crítico", value=clientes_risco_alto, delta="Atenção Necessária")
with col3:
    st.metric(label="Impacto Financeiro Previsto", value=f"R$ {impacto_estimado:,.2f}")

st.markdown("---")

# Seções lado a lado: Gráfico de Sazonalidade e Filtros rápidos
col_esq, col_dir = st.columns([2, 1])

with col_esq:
    st.markdown("### 📅 Projeção de Risco por Mês (Próximo Semestre)")
    # Simulando a tendência com base na lógica de negócio e comportamento dos atrasos
    dados_sazonalidade = pd.DataFrame({
        'Mês': ['Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'],
        'Risco Estimado (%)': [4.0, 8.0, 15.0, 10.0, 18.0, 12.0]
    })
    st.bar_chart(data=dados_sazonalidade, x='Mês', y='Risco Estimado (%)', color='#ff4b4b')

with col_dir:
    st.markdown("### 🔍 Regras de Negócio Aplicadas")
    st.info("**Atrasos Alternados:** O modelo identifica o padrão de 'paga um mês, atrasa o outro' como comportamento de risco moderado a alto, antecipando a quebra de contrato.")
    st.success("**Ação Recomendada:** Entrar em contato com o cliente 3 dias antes do vencimento nos meses sinalizados em vermelho.")

st.markdown("---")

# Tabela Detalhada com formatação condicional
st.markdown("### 📋 Lista Geral de Clientes e Probabilidade Próxima")

# Função para colorir o risco e facilitar a leitura dos coordenadores
def colorir_risco(val):
    if val > 0.60:
        return 'background-color: #ffcccc; color: #cc0000; font-weight: bold;'
    elif val > 0.25:
        return 'background-color: #fff3cd; color: #856404;'
    else:
        return 'background-color: #d4edda; color: #155724;'

# Formatando para exibição em porcentagem amigável
df_exibicao = df_original.copy()
df_exibicao['Probabilidade_Inadimplencia'] = df_exibicao['Probabilidade_Inadimplencia']

# Renderizando a tabela interativa na tela
st.dataframe(
    df_exibicao.style.map(colorir_risco, subset=['Probabilidade_Inadimplencia']).format({'Probabilidade_Inadimplencia': '{:.1%}'}),
    use_container_width=True
)
