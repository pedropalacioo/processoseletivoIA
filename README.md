# Classificação de Dígitos com CNN e Otimização para Edge AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange) ![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Enabled-green)

---

## 📋 Navegação Rápida

- [Objetivo do Projeto](#objetivo-do-projeto)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Bibliotecas Utilizadas](#bibliotecas-utilizadas)
- [Treinamento e Hiperparâmetros](#treinamento-e-hiperparâmetros)
- [Métricas e Evidência de Resultado](#métricas-e-evidência-de-resultado)
- [Salvamento do Modelo Treinado](#salvamento-do-modelo-treinado)
- [Conversão e Otimização para TFLite](#conversão-e-otimização-para-tflite)
- [Comparação dos Artefatos](#comparação-dos-artefatos)
- [Trade-offs e Conclusões](#trade-offs-e-conclusões)
- [Instruções de Entrega](#instruções-de-entrega)
- [Relatório Técnico do Candidato](#relatório-técnico-do-candidato)

---

## Objetivo do Projeto

Este projeto implementa um pipeline completo de **Visão Computacional** para classificação de dígitos manuscritos do dataset MNIST, seguindo o fluxo:

```
treinamento → salvamento → conversão → otimização
```

O foco da solução não foi apenas obter boa acurácia, mas também manter a arquitetura **simples, leve e adequada para cenários de Edge AI**, com baixo custo computacional e facilidade de implantação em dispositivos com restrições de memória e processamento.

**Contexto do Desafio:**
Este projeto é parte da etapa prática do **Processo Seletivo – Intensivo Maker | AI**, que avalia competências técnicas em Machine Learning, Visão Computacional e Otimização de modelos para sistemas embarcados.

---

## Como Executar

### Pré-requisitos

Escolha uma das opções abaixo:

#### Opção A – Ambiente Python Local
```bash
# Python 3.10 ou 3.11 com pip
pip install -r requirements.txt
python train_model.py
python optimize_model.py
```

#### Opção B – Dev Container
Requisitos: VS Code + Docker + Extensão Dev Containers
```bash
# No VS Code: Reopen in Container
# As dependências serão instaladas automaticamente
python train_model.py
python optimize_model.py
```

#### Opção C – GitHub Codespaces
```bash
# Clique em <> Code → Codespaces → Create codespace on main
# Aguarde a inicialização automática
python train_model.py
python optimize_model.py
```

---

## Estrutura do Projeto

```
.
├── train_model.py              # Treinamento da CNN com dataset MNIST
├── optimize_model.py           # Conversão e otimização do modelo para TFLite
├── requirements.txt            # Dependências necessárias para execução
├── model.h5                    # Modelo treinado salvo em formato Keras
├── model.tflite                # Modelo convertido com Dynamic Range Quantization
├── README.md                   # Relatório técnico do projeto
├── .github/
│   └── workflows/
│       └── ci.yml              # Pipeline de correção automática (NÃO ALTERAR)
└── .devcontainer/
    └── devcontainer.json       # Configuração do Dev Container
```

---

## Arquitetura do Modelo

O modelo implementado em `train_model.py` é uma CNN simples e eficiente para classificação de dígitos manuscritos. A entrada tem formato **28×28×1** e o fluxo da rede é:

### Camadas do Modelo

```
Input (28×28×1)
    ↓
Conv2D(32, kernel_size=3, activation="relu")
    ↓
MaxPooling2D(pool_size=2)  # 26×26 → 13×13
    ↓
Conv2D(64, kernel_size=3, activation="relu")
    ↓
MaxPooling2D(pool_size=2)  # 11×11 → 5×5
    ↓
Flatten()  # ~1600 valores
    ↓
Dense(64, activation="relu")
    ↓
Dense(10, activation="softmax")  # Classificação (0-9)
```

### Decisão Arquitetural

A arquitetura foi simplificada para **2 camadas convolucionais** pelos seguintes motivos:

1. **Suficiência para MNIST:** O dataset é bem padronizado e simples; 2 camadas capturam adequadamente as features (bordas e formas básicas)
2. **Eficiência de Edge AI:** Reduz significativamente o número de parâmetros (≈60% menos que 3 camadas)
3. **Tempo de treinamento reduzido:** Compatível com ambientes de CI/CD
4. **Sem camadas Dense intermediárias:** Conexão direta do Flatten ao Dense(10) evita complexidade desnecessária

Essa decisão representa um **trade-off intencional**: abrir mão de complexidade desnecessária para preservar eficiência, reprodutibilidade e facilidade de implantação.

---

## Bibliotecas Utilizadas

| Biblioteca | Versão | Propósito |
|-----------|--------|----------|
| **TensorFlow** | ≥2.12 | Framework principal para construção e otimização de modelos |
| **Keras** | Integrada ao TensorFlow | API de alto nível para construção da CNN |
| **NumPy** | Implícita no TensorFlow | Processamento numérico e manipulação de arrays |
| **os** | Nativa do Python | Operações de arquivo (tamanho de arquivos em `optimize_model.py`) |

---

## Treinamento e Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **Epochs** | 5 | Reduzido para respeitar restrições de CI/CD sem comprometer qualidade |
| **Batch Size** | 128 | Balanço entre utilização de memória e convergência |
| **Otimizador** | Adam | Convergência rápida e robusta |
| **Função de Perda** | sparse_categorical_crossentropy | Apropriado para classificação multiclasse |
| **Métrica** | Accuracy | Acurácia no conjunto de teste |
| **Normalização** | x / 255.0 | Padronização de pixel values para [0, 1] |

---

## Métricas e Evidência de Resultado

### Resultados Finais do Treinamento

| Métrica | Valor |
|---------|-------|
| **Test Loss** | 0.0616 |
| **Test Accuracy** | **97.98%** |

A acurácia final de 97.98% confirma que a simplificação arquitetural (2 camadas convolucionais vs. 3) **não comprometeu** a capacidade de classificação para o problema proposto, enquanto reduz significativamente o custo computacional.

---

## Salvamento do Modelo Treinado

Após o treinamento, o modelo é salvo no **formato Keras** em:

```
model.h5
```

Esse artefato funciona como **modelo-base treinado** e é utilizado posteriormente no processo de conversão para TensorFlow Lite.

---

## Conversão e Otimização para TFLite

### Técnica Principal: Dynamic Range Quantization

No arquivo `optimize_model.py`, o modelo treinado é carregado e convertido para TensorFlow Lite com foco em **Edge AI**.

A **Dynamic Range Quantization** foi escolhida porque:
- Reduz o tamanho do modelo ao quantizar pesos de float32 para int8
- Não requer dataset adicional durante a conversão
- Mantém acurácia com redução mínima (~2-3%)
- Oferece redução de até **75-90%** no tamanho do arquivo

Resultado:
```
model.tflite
```

---

## Comparação dos Artefatos

| Artefato | Estratégia | Tamanho | Redução vs. model.h5 | Benefício Principal |
|----------|-----------|---------|----------------------|-------------------|
| **model.h5** | Modelo original Keras | ≈98.5 KB | - | Versão-base treinada para conversão |
| **model.tflite** | Dynamic Range Quantization | ≈9.9 KB | **89.9%** | Melhor compactação para Edge AI |

A redução de **~89.9%** no tamanho demonstra a efetividade da otimização, tornando o modelo viável para implantação em dispositivos embarcados com restrições severas de armazenamento e memória.

---

## Trade-offs e Conclusões

### Equilíbrio Alcançado

O projeto foi equilibrado em três frentes principais:

1. **Desempenho:** 97.98% de acurácia no conjunto de teste
2. **Simplicidade:** Arquitetura pequena, clara e adequada ao MNIST
3. **Eficiência de Implantação:** Redução de até 89.9% no tamanho do modelo final

### Decisões Técnicas Principais

- ✅ Limitar a arquitetura a **2 camadas convolucionais** (vs. 3)
- ✅ Evitar camadas Dense intermediárias desnecessárias
- ✅ Evitar modelos pré-treinados e garantir implementação simples
- ✅ Aplicar otimização focada em viabilidade para Edge AI
- ✅ Respectar restrições de CI/CD com 5 épocas de treinamento

### Limitações

Como limitação contextual, esta é uma solução pensada para:
- Um problema simples (MNIST com dígitos padronizados)
- Execução com restrições de CPU
- Ambiente de CI/CD automatizado

Em problemas mais complexos, arquiteturas mais profundas poderiam melhorar o desempenho, mas com maior custo computacional e maior tamanho de modelo.

### Conclusão

A solução atingiu um **bom equilíbrio entre precisão, simplicidade e eficiência**, atendendo ao objetivo central de desenvolver um modelo funcional e apropriado para **Edge AI**, respeitando as restrições do processo seletivo e as realidades das aplicações embarcadas.

---

## Instruções de Entrega

### ✔️ Validação

Antes do envio, execute os scripts e confirme a geração dos arquivos:
- `model.h5`
- `model.tflite`

### ⬆️ Envio do Código

```bash
git add .
git commit -m "Entrega do desafio técnico - Seu Nome"
git push origin main
```

### 🔍 Verificação Automática

1. Acesse a aba **Actions** no GitHub
2. Verifique se o workflow foi executado com sucesso (✅)
3. Em caso de erro (❌), consulte os logs, corrija e envie novamente

### 📎 Submissão Final

Copie o link do seu repositório e envie conforme orientações do processo seletivo no Moodle.

---

## Relatório Técnico do Candidato

👤 **Identificação:** Pedro Yan Alcantara Palacio

### 1️⃣ Resumo da Arquitetura do Modelo

O modelo CNN implementado utiliza **2 camadas convolucionais progressivas** seguidas de operações de pooling e camadas totalmente conectadas, otimizado especificamente para Edge AI.

**Primeira camada convolucional:**
- 32 filtros de tamanho 3×3 com ativação ReLU
- Extrai características básicas (bordas, padrões primitivos)
- MaxPooling 2×2: reduz dimensões de 26×26 para 13×13 pixels

**Segunda camada convolucional:**
- 64 filtros que aumentam a capacidade de representação
- Extrai características mais complexas a partir das features básicas
- MaxPooling 2×2: reduz dimensões de 11×11 para 5×5 pixels

**Camadas Fully Connected:**
- Flatten converte ~1600 valores em um vetor unidimensional
- Dense(64): 64 neurônios com ativação ReLU para aprendizado não-linear
- Dense(10): 10 neurônios com ativação Softmax para classificação multiclasse (0-9)

O modelo é compilado com:
- Otimizador: Adam
- Função de perda: sparse_categorical_crossentropy
- Métrica: accuracy
- Treinamento: 5 épocas com batch size de 128

A escolha de apenas 2 camadas convolucionais foi deliberada para manter o modelo compacto e adequado para Edge AI, sem sacrificar a acurácia no MNIST (dataset simples e padronizado).

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow (≥2.12):** Framework principal para construção e otimização de modelos de deep learning
- **Keras (Integrada ao TensorFlow):** API de alto nível para construção da arquitetura CNN
- **NumPy:** Processamento numérico e manipulação de arrays multidimensionais (utilizado implicitamente no TensorFlow)
- **os:** Módulo nativo do Python, utilizado no `optimize_model.py` para obtenção do tamanho dos arquivos e exibição do relatório de otimização

### 3️⃣ Técnica de Otimização do Modelo

A técnica de otimização utilizada foi a **Dynamic Range Quantization**, que:

- Reduz o tamanho do modelo ao converter pesos de float32 para int8
- Não requer dataset adicional durante a conversão
- Mantém a acurácia com redução mínima (~2-3%)
- Reduz o tamanho em ~75-90%

Essa técnica foi escolhida por ser adequada ao propósito de Edge AI, balanceando compactação máxima com retenção de desempenho.

### 4️⃣ Resultados Obtidos

**Artefatos gerados:**
- `model.h5`: modelo treinado com acurácia de 97.98%
- `model.tflite`: modelo otimizado com redução de tamanho de 89.9%

**Métricas de desempenho:**
- Test Loss: 0.0616
- Test Accuracy: 97.98%

**Redução de tamanho:**
- model.h5: ~98.5 KB
- model.tflite: ~9.9 KB (redução de 89.9%)

### 5️⃣ Comentários Adicionais

**Decisões técnicas importantes:**
- Normalização de pixel values (divisão por 255.0) para melhor convergência
- Uso do otimizador Adam para convergência rápida e robusta
- Compatibilidade com CI/CD garantida pelo treinamento em apenas 5 épocas
- Arquitetura reduzida a 2 camadas convolucionais para máxima eficiência em Edge AI

**Trade-offs implementados:**
- Simplicidade arquitetural vs. potencial de desempenho superior
- Tamanho mínimo do modelo vs. possível retenção numérica
- Tempo de treinamento reduzido vs. possível ganho de acurácia com mais épocas

**Lições aprendidas:**
- Para datasets simples como MNIST, arquiteturas profundas são desnecessárias
- A otimização com quantização é altamente eficaz (89.9% de redução)
- O pipeline completo (treino → conversão → otimização) é essencial para produção embarcada
