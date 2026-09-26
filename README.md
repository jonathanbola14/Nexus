# Nexus

Assistente de voz pessoal executado pelo terminal. O Nexus detecta a palavra de ativação, transcreve a fala em português, consulta um agente baseado em NVIDIA NIM e reproduz a resposta com síntese de voz em pt-BR.

O processamento de captura, reconhecimento de fala e síntese de voz é feito localmente. A resposta do agente usa a API NVIDIA e requer uma chave de acesso.

## Requisitos

- Linux x86_64 e Python 3.12.
- [uv](https://docs.astral.sh/uv/) para instalar e executar o ambiente do projeto.
- Microfone e saída de áudio disponíveis no sistema.
- Bibliotecas de áudio do sistema necessárias para compilar ou executar PyAudio, incluindo PortAudio.
- Uma chave de API NVIDIA com acesso ao modelo configurado no agente.

O PyTorch está fixado a uma wheel CPU para Python 3.12 em Linux x86_64 no `pyproject.toml`.

## Instalação

Na raiz do projeto, instale as dependências:

```bash
uv sync
```

Configure a chave da API no arquivo `.env` na raiz do projeto:

```dotenv
NVIDIA_API_KEY=sua-chave-nvidia
```

Mantenha esse arquivo privado e não publique nem compartilhe sua chave.

## Execução

Para iniciar com detecção da palavra de ativação:

```bash
uv run python main.py
```

Para iniciar a captura de fala sem esperar pela palavra de ativação:

```bash
uv run python main.py --sem-ativacao
```

Encerre com `Ctrl+C`. Na primeira execução, se `audios/noise.wav` não existir, o programa grava alguns segundos de ruído ambiente para usar na redução de ruído. Fique em silêncio durante essa gravação.

## Fluxo

1. `WakeWord.py` detecta a palavra de ativação usando `models/nexus.onnx`. Com `--sem-ativacao`, essa etapa é ignorada.
2. `utils/recorder.py` captura a fala e encerra a gravação após detectar silêncio.
3. `STT.py` reduz o ruído e transcreve o áudio com o modelo Parakeet ONNX, configurado para português do Brasil.
4. `LLM.py` envia a mensagem ao agente NVIDIA, que inclui ferramentas para consultar data e hora.
5. `main.py` recebe a resposta em streaming e separa as frases para que possam ser sintetizadas enquanto o restante da resposta é gerado.
6. O áudio é sintetizado pelo Kokoro e reproduzido na saída padrão do sistema.

## Arquivos necessários

- `models/nexus.onnx`, `models/embedding_model.onnx` e `models/melspectrogram.onnx` para a palavra de ativação.
- `audios/activation.wav` e `audios/beep.wav` para os sinais sonoros.
- `audios/noise.wav` como perfil de ruído; é criado na primeira execução se estiver ausente.

O projeto depende de dispositivos de áudio reais e não está preparado para execução sem microfone ou em ambientes headless. Não há suíte de testes automatizada configurada atualmente.