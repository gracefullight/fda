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


def plot_series(series: pd.DataFrame, series_name: str, series_index: int) -> None:
    """Plot a pandas Series with custom title and index."""
    # seaborn의 Dark2 색상 팔레트를 리스트로 변환 (각 species별로 다른 색상 사용)
    palette = list(sns.palettes.mpl_palette("Dark2"))
    # series DataFrame에서 x축 데이터(Row 컬럼) 추출
    xs = series["Row"]
    # series DataFrame에서 y축 데이터(Sepal.Length 컬럼) 추출
    ys = series["Sepal.Length"]

    # matplotlib plot 함수로 선 그래프 그리기
    # (label: 범례에 표시될 이름, color: 팔레트에서 index에 해당하는 색상)
    plt.plot(xs, ys, label=series_name, color=palette[series_index])


def iris_series(df: pd.DataFrame) -> None:
    """Plot each species as a separate series."""
    # matplotlib subplots로 figure와 axes 객체 명시적 생성
    # (figsize: 그래프 크기, layout="constrained": 자동 레이아웃 조정)
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    # DataFrame을 Row 컬럼 기준으로 오름차순 정렬
    df_sorted = df.sort_values("Row", ascending=True)
    # Species별로 그룹화하여 각 그룹에 대해 반복 (enumerate로 인덱스도 함께 추출)
    for i, (series_name, series) in enumerate(df_sorted.groupby("Species")):
        # 각 species 그룹에 대해 plot_series 함수 호출
        plot_series(series, series_name, i)
        # figure에 범례 추가 (title: 범례 제목, bbox_to_anchor: 범례 위치, loc: 범례 정렬)
        fig.legend(title="Species", bbox_to_anchor=(1, 1), loc="upper left")
    # seaborn의 despine 함수로 상단/우측 테두리 제거 (fig, ax 명시적 전달)
    sns.despine(fig=fig, ax=ax)
    plt.xlabel("Row")
    plt.ylabel("Sepal.Length")
    plt.show()
