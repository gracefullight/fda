from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def load_iris_data() -> pd.DataFrame:
    """Load iris dataset from assets directory."""
    current_dir = Path(__file__).parent
    assets_path = current_dir / "assets" / "iris.csv"
    return pd.read_csv(assets_path)


def iris_head(df: pd.DataFrame, n: int = 5) -> None:
    """Display the first n rows of iris dataset."""
    print(df.head(n))  # noqa: T201


def iris_scatter_pandas(df: pd.DataFrame) -> None:
    """pandas의 기본 plot으로 산점도 생성."""
    # pandas DataFrame의 plot 메소드 - 기본 scatter plot 생성
    df.plot(kind="scatter", x="Sepal.Width", y="Petal.Length", s=32, alpha=0.8, figsize=(10, 6))
    plt.title("Sepal.Width vs Petal.Length (pandas plot)")
    plt.xlabel("Sepal.Width")
    plt.ylabel("Petal.Length")
    # 상단/우측 테두리 제거
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.show()


def iris_scatter_seaborn(df: pd.DataFrame) -> None:
    """seaborn으로 species별 색상 구분된 산점도 생성"""
    plt.figure(figsize=(10, 6))
    # seaborn의 scatterplot 예쁜 산점도 생성
    sns.scatterplot(data=df, x="Sepal.Width", y="Petal.Length", hue="Species", s=50, alpha=0.8)
    plt.title("Sepal.Width vs Petal.Length by Species (seaborn)")
    plt.xlabel("Sepal.Width")
    plt.ylabel("Petal.Length")
    # 상단/우측 테두리 제거
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.show()


def iris_scatter(df: pd.DataFrame) -> None:
    """pandas와 seaborn 산점도 모두 보여주기"""
    print("1. pandas 기본 plot:")  # noqa: T201
    iris_scatter_pandas(df)

    print("2. seaborn plot (species별 색상 구분):")  # noqa: T201
    iris_scatter_seaborn(df)
