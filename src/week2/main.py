from pathlib import Path

import pandas as pd

current_dir = Path(__file__).parent.parent


def load_iris_data() -> pd.DataFrame:
    """Load iris dataset from assets directory."""
    assets_path = current_dir / "assets" / "iris-u2.csv"
    return pd.read_csv(assets_path)


def run_week2() -> None:
    """Run week 2 tasks."""
    iris_data = load_iris_data()

    # 단일 대괄호 []: Series 반환 (1차원 데이터)
    # - 컬럼 하나를 선택할 때 사용
    # - 반환 타입: pandas.Series
    # - 인덱스와 값만 포함된 1차원 구조
    print("=== Series (1차원) ===")
    print(type(iris_data["Sepal.Length"]))  # <class 'pandas.core.series.Series'>
    print(iris_data["Sepal.Length"])

    print("\n=== DataFrame (2차원) ===")
    # 이중 대괄호 [[]]: DataFrame 반환 (2차원 데이터)
    # - 컬럼 하나 또는 여러 개를 선택할 때 사용
    # - 반환 타입: pandas.DataFrame
    # - 행과 열 구조를 유지한 2차원 테이블 형태
    print(type(iris_data[["Sepal.Length"]]))  # <class 'pandas.core.frame.DataFrame'>
    print(iris_data[["Sepal.Length"]])

    print("\n=== 데이터 인덱싱 방법들 ===")

    # 1. 슬라이싱 [start:end]: 행(row) 기준 슬라이싱
    # - 정수 인덱스 기준으로 행을 선택
    # - end는 포함되지 않음 (0:2 = 0, 1번째 행만)
    print("\n1. 기본 슬라이싱 [0:2] - 0, 1번째 행:")
    print(iris_data[0:2])

    # 2. .loc[행, 열]: 라벨 기반 인덱싱
    # - 행: 인덱스 라벨로 선택 (3:5는 3,4,5 모두 포함)
    # - 열: 컬럼명으로 선택
    # - 장점: 직관적이고 명확한 라벨 사용
    print("\n2. .loc[3:5, 컬럼명] - 라벨 기반 선택:")
    print(iris_data.loc[3:5, ["Sepal.Length", "Species"]])

    # 3. .iloc[행, 열]: 위치 기반 인덱싱 (정수만 사용)
    # - 행: 정수 위치로 선택 (3:6은 3,4,5번째 행)
    # - 열: 정수 위치로 선택 ([1, 5] = 2번째, 6번째 컬럼)
    # - 장점: 컬럼명을 몰라도 위치만으로 선택 가능
    print("\n3. .iloc[3:6, [1, 5]] - 위치 기반 선택:")
    print(iris_data.iloc[3:6, [1, 5]])

    iris_data_out = iris_data.drop(["Row"], axis=1)
    print(iris_data_out)

    iris_data_out.to_csv(current_dir / "assets" / "iris-u2-out.csv")
