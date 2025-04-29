# pipeline_reclamacoes_publicas_mar_2025.py
# Autor: Marcos Vinicius do Couto Scatolino
# Projeto: Análise Estratégica de Reclamações Públicas no setor de Logística e E-commerce

# ==========================================
# IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# ==========================================

# Bibliotecas essenciais para manipulação de dados, visualização, NLP e machine learning

import os
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# BLOCO 1 – Leitura e Limpeza dos Dados
# ==========================================

# Definir o caminho do arquivo
pasta = 'dados'
nome_arquivo = 'base_completa_2025-03.csv'
caminho_dados = os.path.join(pasta, nome_arquivo)

# Verifica se o arquivo existe
if not os.path.exists(caminho_dados):
    raise FileNotFoundError(f'Arquivo não encontrado: {caminho_dados}')

# Carrega o CSV com codificação original
df = pd.read_csv(caminho_dados, sep=';', encoding='latin-1')

# Função para limpar e padronizar os nomes das colunas
def limpar_colunas(col):
    col = col.encode('latin1').decode('utf-8')
    col = col.replace(' ', '_')
    col = col.replace('ã', 'a').replace('á', 'a').replace('â', 'a').replace('ê', 'e')
    col = col.replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ô', 'o')
    col = col.replace('ç', 'c').replace('ú', 'u').replace('Ã', 'A').replace('Õ', 'O')
    col = col.replace('Ê', 'E').replace('Á', 'A').replace('É', 'E')
    col = col.lower()
    return col

# Aplica a limpeza nos nomes das colunas
df.columns = [limpar_colunas(c) for c in df.columns]
df.rename(columns={df.columns[0]: 'gestor'}, inplace=True)

# 🔧 LIMPEZA EXTRA: corrigir acentuação nos conteúdos das células (valores)
for coluna in df.select_dtypes(include='object').columns:
    df[coluna] = df[coluna].apply(lambda x: x.encode('latin1').decode('utf-8') if isinstance(x, str) else x)

# ===============================
# BLOCO 2 – Foco: logística e e-commerce
# ===============================

# 1. Empresas com mais reclamações
print('\n📌 Empresas com mais reclamações:')
print(df['nome_fantasia'].value_counts().head(20))

# 2. Segmentos de mercado disponíveis
print('\n📌 Segmentos de mercado presentes:')
print(df['segmento_de_mercado'].value_counts().head(20))

# 3. Filtro de empresas de logística/e-commerce
palavras_chave = [
    'logistica', 'entrega', 'e-commerce', 'transportadora',
    'correios', 'mercado livre', 'magalu', 'amazon'
]

filtro = df['segmento_de_mercado'].str.lower().str.contains('|'.join(palavras_chave), na=False) | \
         df['nome_fantasia'].str.lower().str.contains('|'.join(palavras_chave), na=False)

df_log = df[filtro].copy()
print(f"\n🔍 Total de registros filtrados (logística/e-commerce): {len(df_log)}")

# 4. Principais problemas relatados
print('\n📌 Principais problemas relatados nas empresas filtradas:')
print(df_log['problema'].value_counts().head(15))

# 5. Nota média x tempo de resposta
df_avaliados = df_log.dropna(subset=['tempo_resposta', 'nota_do_consumidor'])

media_resposta_nota = df_avaliados.groupby('nome_fantasia')[
    ['tempo_resposta', 'nota_do_consumidor']
].mean().sort_values(by='nota_do_consumidor', ascending=False)

print('\n📊 Empresas com melhor média de nota (avaliadas):')
print(media_resposta_nota.head(10).round(2))

print('\n📉 Empresas com pior média de nota (avaliadas):')
print(media_resposta_nota.tail(10).round(2))

# ===============================
# BLOCO 3 – Visualizações dos dados
# ===============================

# Estilo atualizado
sns.set_theme(style='whitegrid')
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 1. Gráfico de Barras – Top 10 Problemas
# ----------------------------
top_problemas = df_log['problema'].value_counts().head(10)

plt.figure(num='Top 10 Problemas Mais Comuns', figsize=(12,6))
sns.barplot(x=top_problemas.values, y=top_problemas.index, palette='crest')
plt.title('Top 10 Problemas Mais Comuns em Logística/E-commerce')
plt.xlabel('Número de Reclamações')
plt.ylabel('Problema')
plt.tight_layout()
plt.show()

# ----------------------------
# 2. Scatterplot – Nota x Tempo de Resposta
# ----------------------------
df_avaliados_plot = df_log.dropna(subset=['tempo_resposta', 'nota_do_consumidor']).copy()

unique_empresas = df_avaliados_plot['nome_fantasia'].unique()
palette = sns.color_palette('tab10', n_colors=len(unique_empresas))
empresa_cor = {nome: palette[i % len(palette)] for i, nome in enumerate(unique_empresas)}

plt.figure(num='Nota x Tempo de Resposta', figsize=(10,6))
sns.scatterplot(
    data=df_avaliados_plot,
    x='tempo_resposta',
    y='nota_do_consumidor',
    hue='nome_fantasia',
    palette=empresa_cor,
    alpha=0.75,
    edgecolor='black'
)
plt.title('Tempo de Resposta x Nota do Consumidor')
plt.xlabel('Tempo de Resposta (dias)')
plt.ylabel('Nota do Consumidor (1 a 5)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Empresa')
plt.tight_layout()
plt.show()

# ----------------------------
# 3. WordCloud – Visual + Frequência
# ----------------------------

# Texto base + limpeza
texto_problemas = ' '.join(df_log['problema'].dropna().tolist())
texto_problemas = (
    texto_problemas.replace("NAO", "não")
                   .replace("Nao", "não")
                   .replace("nÃ£o", "não")
                   .replace("nAo", "não")
                   .lower()
)
texto_problemas = re.sub(r'[^\w\s]', '', texto_problemas)

# Quebra palavras + remoção de stopwords
palavras = texto_problemas.split()
stopwords = ['de', 'o', 'a', 'e', 'com', 'para', 'por', 'em', 'do', 'na', 'no', 'da',
             'ao', 'os', 'as', 'dos', 'das', 'se']
palavras_filtradas = [p for p in palavras if p not in stopwords and len(p) > 1]

# Frequência e preview
frequencia_palavras = Counter(palavras_filtradas)
print('\n🔠 Top 50 palavras mais citadas:')
for palavra, freq in frequencia_palavras.most_common(50):
    print(f'{palavra}: {freq}')

# WordCloud com nome personalizado na janela
texto_filtrado = ' '.join(palavras_filtradas)
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    colormap='Dark2',
    max_words=100
).generate(texto_filtrado)

plt.figure(num='WordCloud – Problemas Mais Citados', figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Mais Citadas nos Problemas Relatados')
plt.tight_layout()
plt.show()

################################
# Clusternização
################################

# 1. Coleta dos textos dos problemas
problemas = df_log['problema'].dropna().values

# 2. Lista de stopwords personalizadas (em português)
stopwords_pt = [
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não',
    'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas',
    'foi', 'ao', 'à', 'pelo', 'pela', 'sobre', 'entre', 'sem', 'também',
    'já', 'tem', 'há', 'é', 'ser', 'está', 'estavam'
]

# 3. Vetorização com TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords_pt, max_features=1000)
X = vectorizer.fit_transform(problemas)

# 4. Clusterização com KMeans
k = 5
modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
modelo_kmeans.fit(X)

# 5. Redução dimensional com PCA para visualização
pca = PCA(n_components=2, random_state=42)
X_reduzido = pca.fit_transform(X.toarray())

# 6. Visualização dos clusters (com legenda)
plt.figure(num='Clusterização de Reclamações com KMeans', figsize=(12, 6))
cores = plt.cm.tab10(np.arange(k))

for i in range(k):
    plt.scatter(
        X_reduzido[modelo_kmeans.labels_ == i, 0],
        X_reduzido[modelo_kmeans.labels_ == i, 1],
        color=cores[i],
        label=f'Cluster {i}',
        alpha=0.7
    )

plt.title('Agrupamento de Reclamações com KMeans (PCA reduzido)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Clusters', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Adiciona os clusters ao DataFrame original
df_log['cluster'] = modelo_kmeans.labels_

# 8. Mostra exemplos reais de cada cluster
print('\n📌 Exemplos de reclamações por cluster:\n')
for i in range(k):
    print(f'\n🧩 Cluster {i}:')
    print(df_log[df_log['cluster'] == i]['problema'].sample(5, random_state=42).to_list())


####
# classificação dos clusters com regressão logística
###

# 1. Dados de entrada: texto dos problemas
problemas = df_log['problema'].dropna()
clusters = df_log.loc[problemas.index, 'cluster']

# 2. Vetorização com o mesmo TF-IDF (reaproveitar ou refazer)
vectorizer = TfidfVectorizer(stop_words=stopwords_pt, max_features=1000)
X = vectorizer.fit_transform(problemas)
y = clusters

# 3. Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Treinar o modelo
modelo_lr = LogisticRegression(max_iter=1000)
modelo_lr.fit(X_train, y_train)

# 5. Avaliar a performance
y_pred = modelo_lr.predict(X_test)

print("\n📊 Classificação – Regressão Logística sobre Clusters")
print(classification_report(y_test, y_pred))
print("📉 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 6. Prever a qual cluster pertence uma nova reclamação
nova_reclamacao = ["O produto chegou danificado e a empresa não responde"]
nova_vetor = vectorizer.transform(nova_reclamacao)
cluster_previsto = modelo_lr.predict(nova_vetor)[0]

print(f"\n🧠 Nova reclamação: \"{nova_reclamacao[0]}\"")
print(f"➡️ Classificada no cluster {cluster_previsto}")

###
# exportação de graficos
###

# Criar pasta de saída, se não existir
os.makedirs('outputs', exist_ok=True)

# 1. Gráfico de barras – Top 10 Problemas
plt.figure(figsize=(12,6))
sns.barplot(x=top_problemas.values, y=top_problemas.index, palette='crest')
plt.title('Top 10 Problemas Mais Comuns em Logística/E-commerce')
plt.xlabel('Número de Reclamações')
plt.ylabel('Problema')
plt.tight_layout()
plt.savefig('outputs/top_problemas.png', dpi=300)
plt.close()

# 2. Gráfico de dispersão – Tempo de resposta x Nota
plt.figure(figsize=(10,6))
for i in range(k):
    plt.scatter(
        X_reduzido[modelo_kmeans.labels_ == i, 0],
        X_reduzido[modelo_kmeans.labels_ == i, 1],
        label=f'Cluster {i}',
        alpha=0.7
    )
plt.title('Clusterização de Reclamações com KMeans (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/clusters_kmeans.png', dpi=300)
plt.close()

# 3. Gráfico de dispersão – Nota x Tempo de Resposta (com legenda)
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df_avaliados_plot,
    x='tempo_resposta',
    y='nota_do_consumidor',
    hue='nome_fantasia',
    palette='tab10',
    alpha=0.75,
    edgecolor='black'
)
plt.title('Tempo de Resposta x Nota do Consumidor')
plt.xlabel('Tempo de Resposta (dias)')
plt.ylabel('Nota do Consumidor (1 a 5)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Empresa')
plt.tight_layout()
plt.savefig('outputs/nota_vs_resposta.png', dpi=300)
plt.close()

# 4. Wordcloud – Palavras mais citadas
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    colormap='Dark2',
    max_words=100
).generate(texto_filtrado)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Mais Citadas nos Problemas Relatados')
plt.tight_layout()
plt.savefig('outputs/wordcloud_problemas.png', dpi=300)
plt.close()

class PDF(FPDF):
    def header(self):
        pass  # sem cabeçalho automático

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')

    def add_section(self, titulo, imagem=None, texto=None):
        # Adiciona quebra de página se não tiver espaço para título + imagem
        altura_prevista = 10 + (90 if imagem else 0) + (30 if texto else 0)
        if self.get_y() + altura_prevista > self.h - 20:
            self.add_page()

        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, titulo, ln=True)

        if imagem and os.path.exists(imagem):
            self.image(imagem, w=170)
            self.ln(5)

        if texto:
            self.set_font("Arial", '', 11)
            self.multi_cell(0, 8, texto, align='J')
            self.ln(5)

###
# PDF
###

def limpar_texto(texto):
    return (
        texto.replace("’", "'")
             .replace("–", "-")
             .replace("—", "-")
             .replace("“", '"')
             .replace("”", '"')
             .replace("•", "-")
             .replace("…", "...")
             .encode("latin-1", errors="ignore")
             .decode("latin-1")
    )

class PDF_ABNT(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(left=30, top=30, right=20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 9)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')

    def add_section(self, titulo, imagem=None, texto=None):
        altura_prevista = 10 + (90 if imagem else 0) + (60 if texto else 0)
        if self.get_y() + altura_prevista > self.h - 20:
            self.add_page()

        self.set_font("Arial", 'B', 12)
        self.multi_cell(0, 10, limpar_texto(titulo))
        self.ln(1)

        if imagem and os.path.exists(imagem):
            largura_disponivel = self.w - self.l_margin - self.r_margin
            self.image(imagem, w=largura_disponivel)
            self.ln(5)

        if texto:
            self.set_font("Arial", '', 11)
            self.multi_cell(0, 8, limpar_texto(texto), align='J')
            self.ln(5)

# Criar pasta de saída
os.makedirs("outputs", exist_ok=True)

# Criar documento PDF com padrão ABNT
pdf = PDF_ABNT()
pdf.add_page()

# Capa
pdf.set_font("Arial", 'B', 14)
pdf.multi_cell(0, 10, limpar_texto("Análise Estratégica de Reclamações Públicas – Setor de Logística e E-commerce (Mar/2025)"))
pdf.set_font("Arial", '', 11)
pdf.cell(0, 10, limpar_texto("Por: Marcos Vinicius do Couto Scatolino | Analista de Dados"), ln=True)
pdf.ln(10)

# Introdução
intro = (
    "Este relatório analítico apresenta os principais pontos de fricção no relacionamento entre consumidores e empresas do setor de "
    "logística e e-commerce, a partir de um banco público de mais de 150 mil registros de reclamações até março de 2025. "
    "Através de técnicas de análise de dados, visualização e machine learning, foi possível extrair padrões valiosos que podem ajudar empresas "
    "a reduzir atritos, elevar sua reputação, otimizar o tempo de resposta e entregar uma experiência de cliente mais inteligente, humana e eficiente."
)
pdf.set_font("Arial", '', 11)
pdf.multi_cell(0, 8, limpar_texto(intro), align='J')
pdf.ln(10)

# Seções com gráficos e análise interpretativa
pdf.add_section(
    "1. Principais Problemas Apontados por Clientes",
    imagem="outputs/top_problemas.png",
    texto=(
        "O gráfico revela que 'Não entrega' e 'Demora na entrega' são os maiores motivos de insatisfação, seguidos por reembolso difícil e venda enganosa. "
        "A recorrência desses temas indica falhas estruturais na jornada logística e no pós-venda. Empresas que buscam reduzir o CAC e ampliar o LTV "
        "devem focar em resolver esses gargalos com prioridade estratégica, integrando dados de SAC, logística e atendimento em tempo real."
    )
)

pdf.add_section(
    "2. Correlação: Tempo de Resposta e Nota do Consumidor",
    imagem="outputs/nota_vs_resposta.png",
    texto=(
        "Empresas que respondem mais rapidamente às reclamações tendem a ter melhores avaliações. Isso demonstra que tempo de resposta é um KPI crítico "
        "não apenas operacional, mas de reputação. O dado reforça a importância de um SAC orientado por dados (SAC 4.0) e fluxos automatizados com supervisão humana, "
        "otimizando o relacionamento e antecipando crises."
    )
)

pdf.add_section(
    "3. Padrões Ocultos: Segmentação Automática via KMeans",
    imagem="outputs/clusters_kmeans.png",
    texto=(
        "Utilizando KMeans, foram agrupadas reclamações por similaridade sem rótulo prévio. Isso permite categorizar de forma inteligente grandes volumes de feedback, "
        "facilitando a priorização de melhorias. Empresas podem usar essa lógica para criar etiquetas automáticas e dashboards gerenciais mais inteligentes, otimizando esforços de UX, logística e suporte."
    )
)

pdf.add_section(
    "4. Análise de Linguagem Natural: Palavras Mais Citadas",
    imagem="outputs/wordcloud_problemas.png",
    texto=(
        "Termos como 'entrega', 'produto', 'reembolso', 'nao' (normalizado) e 'cancelamento' dominam a percepção dos clientes. Esses pontos evidenciam que, para além de falhas operacionais, "
        "há um desalinhamento entre promessa e entrega, impactando diretamente a confiança. A análise textual reforça a urgência de comunicação transparente, fluxos claros e escuta ativa."
    )
)

pdf.add_section(
    "5. Conclusões Estratégicas e Recomendações",
    texto=(
        "Empresas que atuam em setores intensivos em logística e relacionamento precisam tratar dados de SAC como fonte primária de inteligência. "
        "Os insights apresentados aqui mostram que a experiência do cliente começa muito antes da entrega, e se prolonga no pós-venda. "
        "Organizações que integram dados, escutam com velocidade e reagem com empatia se posicionam à frente da concorrência. "
        "Recomenda-se implementar sistemas de triagem automática com base em clusterização, priorização por sentimento e automatização de contato inicial, com supervisão analítica contínua."
    )
)

# Salvar PDF
pdf.output("outputs/relatorio_final_abnt.pdf")
