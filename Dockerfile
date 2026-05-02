# Jetson AGX Orin (JetPack 6.x, L4T R36.4) で panda-gym + Stable-Baselines3 を動かす環境
#
# ベース: dustynv/pytorch:2.7-r36.4.0 (PyTorch 2.7 + CUDA 12.6 + Python 3.10)
#
# panda-gym (PyBullet ベース) はPyPI に aarch64 wheel が無いため、
# コンテナ内で setup.py が source build される。Linux + gcc では
# Apple Silicon Mac で発生した同梱 zlib のマクロ衝突は起きない。

FROM dustynv/pytorch:2.7-r36.4.0

ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple/ \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 1) 汎用ツール
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2) 動画エンコード（imageio[ffmpeg] のバックエンド）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3) PyBullet をソースビルドするための C/C++ ツールチェーン
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# 4) PyBullet を headless で描画するための OpenGL 系ライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libegl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# 5) Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 6) panda-gym 周辺パッケージ
#    - PyBullet は source build される（aarch64 wheel が PyPI に無い）
#    - PyTorch は base image 同梱なので入れない
RUN pip install --no-cache-dir \
        "panda-gym==3.0.7" \
        "stable-baselines3==2.7.0" \
        "gymnasium==0.29.1" \
        "tensorboard==2.16.2" \
        "imageio[ffmpeg]==2.34.1"

# 7) ワークスペース
WORKDIR /workspace/panda-gym-sb3-rl-starter

# 8) スモーク用デフォルトコマンド: バージョン確認
CMD ["python3", "-c", "import panda_gym, gymnasium, stable_baselines3, torch; print('panda_gym:', panda_gym.__version__); print('gymnasium:', gymnasium.__version__); print('SB3:', stable_baselines3.__version__); print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"]
