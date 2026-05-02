# panda-gym × Stable-Baselines3 SAC + HER on Jetson AGX Orin

7 自由度の Franka Panda アームを、sparse reward の状態で目標位置に到達させる強化学習パイプラインです。
シミュレータは [panda-gym](https://github.com/qgallouedec/panda-gym)（PyBullet ベース）、
学習ライブラリは [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) の SAC + Hindsight Experience Replay。

全工程を Jetson AGX Orin 上の Docker コンテナで完結させます。クラウド不要、AWS 利用費 0 円。

## 結果（PandaReach-v3、25 000 step）

| 項目 | 値 |
|---|---|
| ハードウェア | Jetson AGX Orin（Orin GPU、CUDA 12.6、JetPack 6.2.1） |
| 学習時間 | 15 分 12 秒 |
| 学習終了時の rolling success rate | 100 % |
| Deterministic eval（50 episode） | 100 %（50/50） |
| Eval mean reward | -1.80 |
| 平均到達ステップ数 | 2.81 |

`videos/eval.mp4` は学習後の policy が 1〜3 step で goal に到達する様子。
`videos/random.mp4` はランダム policy がアームをでたらめに動かす比較用動画。

## デモ動画（YouTube）

[![panda-gym × Stable-Baselines3 SAC + HER demo](https://img.youtube.com/vi/E1dQemO-mn0/maxresdefault.jpg)](https://www.youtube.com/watch?v=E1dQemO-mn0)

左：ランダム policy／右：SAC + HER で学習後の deterministic policy（PandaReach-v3）。

## ディレクトリ構成

```
panda-gym-sb3-rl-starter/
├── Dockerfile             # dustynv/pytorch:2.7-r36.4.0 ベース + panda-gym + SB3
├── requirements.txt
├── smoke_test.py           # PandaReach-v3 動作確認 → PNG
├── train.py               # SAC + HER 学習（デフォルト 25 000 step）
├── eval.py                # Deterministic 評価 + MP4
├── demo_random.py         # 学習前比較用のランダム policy MP4
├── plot_curves.py         # TensorBoard ログから run ごとに 5 metric を 1 枚にまとめた学習曲線 PNG を生成
├── plot_compare.py        # TensorBoard ログから success_rate + ep_rew_mean を縦 2 段にまとめた比較 PNG を生成
├── README.md
└── README.ja.md
```

## セットアップ

### 前提

- JetPack 6.2.1（L4T R36.4）+ Docker + nvidia runtime が入った Jetson AGX Orin
- もしくは Docker が使える aarch64 Linux マシン（CUDA は任意）

### Clone

```bash
git clone https://github.com/<your-account>/panda-gym-sb3-rl-starter.git
cd panda-gym-sb3-rl-starter
```

### Docker image をビルド

```bash
docker build -t panda-gym-sb3-rl-starter:latest .
```

初回は `dustynv/pytorch:2.7-r36.4.0`（約 10 GB）の pull と、PyBullet のソースビルドが
コンテナ内で走ります（PyPI に aarch64 wheel が無いため）。Jetson AGX Orin で約 10〜15 分。

### コンテナ起動

学習は約 15 分かかります。SSH が切れてもコンテナが落ちないよう、`sleep infinity` で
daemon 起動しておき、後から `docker exec` で対話シェルに入る運用にします。

```bash
docker run -d --name panda_gym_sb3 \
  --runtime nvidia \
  --network host \
  -v "$(pwd):/workspace/panda-gym-sb3-rl-starter" \
  -w /workspace/panda-gym-sb3-rl-starter \
  panda-gym-sb3-rl-starter:latest \
  sleep infinity

docker exec -it panda_gym_sb3 bash
```

停止と削除：

```bash
docker stop panda_gym_sb3 && docker rm panda_gym_sb3
```

## ワークフロー

### 1. 動作確認

```bash
python3 smoke_test.py
```

`PandaReach-v3` をロードしてランダム行動 50 step、`smoke_first.png` / `smoke_last.png` を保存。
Franka Panda アームと緑色の goal が描画されていれば OK。

### 2. 学習

```bash
python3 train.py
```

デフォルト：`PandaReach-v3`、25 000 timesteps、SAC + `HerReplayBuffer`、
`net_arch=[64, 64]`、`n_critics=1`。モデルは `logs/sac_her_pandareach.zip` に保存。
TensorBoard ログは `logs/tb/`。

### 3. 評価

```bash
python3 eval.py --episodes 50 --video-episodes 20
```

50 個のランダム seed で deterministic rollout、success rate と mean reward を表示し、
`videos/eval.mp4` に動画保存。

### 4. ランダム policy 比較

```bash
python3 demo_random.py --episodes 10
```

「学習前」のベースライン動画を `videos/random.mp4` に保存。

### 5. 学習曲線

```bash
python3 plot_curves.py
```

`logs/tb/` 配下の TensorBoard run ごとに、success rate / episode reward / critic loss /
actor loss / entropy coefficient を縦 5 段に並べた 1 枚の PNG を `plots/` に保存。

### 6. アブレーション：HER 無しで学習

```bash
python3 train.py --no-her --out logs/sac_pandareach_noher
```

同じ SAC + 同じネットワークで、HER を SB3 デフォルトの `DictReplayBuffer` に置換。
goal-conditioned sparse reward タスクで HER がどれくらい効いているかを示す対照実験。

### 7. HER on/off の比較プロット

```bash
python3 plot_compare.py --runs SAC_1:HER SAC_2:no-HER --name reach_her_vs_noher
```

2 つの TensorBoard run の success_rate（上段）と ep_rew_mean（下段）を縦 2 段に
並べた 1 枚の PNG を `plots/` に出力。各 subplot に 2 run の曲線を重ねて表示します。
`SAC_1`（HER on、ステップ 2 で生成）と `SAC_2`（HER off、ステップ 6 で生成）の
両方が `logs/tb/` 配下に存在する必要があります。ブログ記事の比較グラフの生成に
使用しています。

## バージョン

| 項目 | バージョン |
|---|---|
| ベースイメージ | `dustynv/pytorch:2.7-r36.4.0` |
| Python | 3.10 |
| PyTorch | 2.7（CUDA 12.6） |
| panda-gym | 3.0.7 |
| stable-baselines3 | 2.7.0 |
| gymnasium | 0.29.1 |
| pybullet | sdist からソースビルド |

## ライセンス

MIT.
