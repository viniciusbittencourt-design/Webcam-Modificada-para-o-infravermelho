# Protótipo de Termografia Infravermelha de Baixo Custo

## Descrição

Este projeto apresenta o desenvolvimento de um sistema experimental de visualização de radiação infravermelha utilizando uma webcam modificada e processamento digital de imagens em Python.

O objetivo é demonstrar conceitos físicos relacionados à transferência de calor, incluindo radiação térmica, condução, convecção e resfriamento de corpos, estabelecendo uma relação com aplicações da termografia infravermelha na área médica.

O sistema utiliza uma câmera convencional adaptada com remoção do filtro infravermelho original, permitindo maior sensibilidade ao infravermelho próximo (NIR). As imagens capturadas são processadas digitalmente para melhorar a visualização dos padrões térmicos através de técnicas de processamento de imagens.

> Observação: este protótipo possui finalidade experimental e educacional. Ele não substitui câmeras termográficas profissionais utilizadas em aplicações clínicas.

---

# Tecnologias utilizadas

* Python
* OpenCV
* NumPy
* Processamento digital de imagens
* Webcam modificada
* Calibração experimental de intensidade

---

# Funcionamento

O sistema realiza:

* Captura de imagens da webcam
* Conversão e tratamento dos dados de imagem
* Realce de contraste utilizando CLAHE
* Aplicação de mapas de falsa cor
* Detecção de regiões de maior intensidade
* Análise de pontos selecionados pelo usuário
* Perfil de intensidade ao longo de uma linha
* Estimativa experimental de temperatura através de calibração

---

# Estrutura do projeto

```
Projeto-Termografia/
│
├── main.py
├── app_state.py
├── calibration.json
├── requirements.txt
├── README.md
│
└── imagens/
```

---

# Controle da aplicação

O arquivo `app_state.py` gerencia o estado da aplicação, incluindo:

* Configuração da câmera
* Resolução do vídeo
* Ajuste de contraste
* Mapa de cores
* Detecção de hotspots
* Pontos de medição
* Perfil de linha
* Controle de FPS

---

# Calibração

O sistema utiliza uma calibração experimental relacionando intensidade do pixel com temperatura.

O arquivo `calibration.json` contém os coeficientes utilizados:

```
Temperatura = a × intensidade + b
```

Coeficientes:

```
a = 1.7846822836302427

b = -152.86178519661468
```

Com ajuste experimental:

```
R² = 0.999347721434815
```

---

# Instalação

Clone o repositório:

```bash
git clone https://github.com/usuario/projeto-termografia.git
```

Entre na pasta:

```bash
cd projeto-termografia
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute:

```bash
python main.py
```

---

# Aplicações educacionais

Este projeto pode ser utilizado para demonstrar:

* Radiação infravermelha
* Transferência de calor
* Sensoriamento óptico
* Processamento de imagens
* Relação entre física e diagnóstico médico

---

# Limitações

A webcam modificada detecta principalmente infravermelho próximo, enquanto câmeras termográficas profissionais utilizam sensores específicos para infravermelho médio e longo, capazes de medir a radiação térmica emitida por corpos em temperatura ambiente.

Portanto, os valores de temperatura apresentados possuem caráter experimental.

---

# Autor

Vinicius Moreschi Bittencourt

Projeto desenvolvido como parte de atividade de extensão em Física Médica.
