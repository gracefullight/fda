from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_iris_data() -> pd.DataFrame:
    """Load iris dataset from assets directory."""
    current_dir = Path(__file__).parent.parent
    assets_path = current_dir / "assets" / "iris.csv"
    return pd.read_csv(assets_path)


def iris_head(df: pd.DataFrame, n: int = 5) -> None:
    """Display the first n rows of iris dataset."""
    df = df.drop(["Row"], axis=1)
    print(df.head(n))


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
    """seaborn으로 species별 색상 구분된 산점도 생성."""
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
    """pandas와 seaborn 산점도 모두 보여주기."""
    print("1. pandas 기본 plot:")
    iris_scatter_pandas(df)

    print("2. seaborn plot (species별 색상 구분):")
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


def iris_label_encoder(df: pd.DataFrame) -> None:
    """Label encode the 'Species' column in the iris dataset."""
    # LabelEncoder 객체 생성
    le = LabelEncoder()
    df = df.drop(["Row"], axis=1)
    # 'Species' 컬럼을 label encoding하여 새로운 'Species_Encoded' 컬럼 생성
    df["Species"] = le.fit_transform(df["Species"])
    # target 변수에 고유한 species 값 저장
    target = df["Species"].unique()
    # target_code 딕셔너리 생성: species 이름을 인덱스로, 고유 번호를 값으로 매핑
    target_code = dict(zip(target, range(len(target)), strict=False))
    print(target_code)

    # 'Species' 컬럼의 값을 target_code 딕셔너리로 매핑
    df["Species"] = df["Species"].apply(lambda x: target_code[x])
    print(df.head(5))

    # 데이터프레임에서 마지막 컬럼을 제외한 나머지 컬럼을 X로, 마지막 컬럼을 y로 분리
    x = df.iloc[:, :-1]
    # 마지막 컬럼 Species를 y로 설정
    y = df.iloc[:, -1].copy()

    # train_test_split을 사용하여 데이터를 학습용과 테스트용으로 분리
    # test_size=0.3: 전체 데이터의 30%를 테스트용으로
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print("X_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", x_test.shape)
    print("y_test:", y_test.shape)

    # RandomForestClassifier 모델 생성 및 학습
    # n_estimators=50: 50개의 결정 트리 사용
    # max_features="sqrt": 각 트리에서 사용할 최대 특성 수를 제곱근으로 설정
    # (Iris 4개 특성 → sqrt(4)=2개씩만 랜덤 선택하여 트리 다양성 증가, 과적합 방지)
    # oob_score=True: Out-Of-Bag 샘플을 사용하여 모델 성능을 자동으로 검증
    # 랜덤포레스트는 각 트리 학습 시마다 중복 허용 랜덤 추출
    # 이 과정에서 일부 샘플은 사용되지 않고 남음 → OOB 샘플
    # OOB 샘플은 모델 검증용으로 활용되어 성능 평가에 사용
    clf = RandomForestClassifier(n_estimators=50, max_features="sqrt", oob_score=True)
    clf.fit(x_train, y_train)

    print(clf)
    print("Feature importances:", clf.feature_importances_)
    print(f"oob score is {clf.oob_score_:.3f}")

    test_idx = 20
    # 테스트용 데이터에서 특정 인덱스의 데이터를 선택
    test_point = x_test.iloc[test_idx]

    # DataFrame 형태로 예측하여 feature name 경고 방지
    # Series를 1행 DataFrame으로 변환 (to_frame().T)
    pred_test = clf.predict(test_point.to_frame().T)
    # 선택한 테스트 데이터에 대한 예측 확률 계산
    pred_test_probs = clf.predict_proba(test_point.to_frame().T)

    # 예측 결과 출력
    print(
        "Testing point\n",
        test_point,
        ";\npredicted as",
        pred_test[0],
        ";\nactually",
        y_test.iloc[test_idx],
        ";\nprobabilities",
        pred_test_probs[0],
    )

    # 전체 테스트 데이터에 대한 예측 수행
    y_pred = clf.predict(x_test)
    # classification_report: 정밀도, 재현율, F1-score 등 상세 성능 지표 출력
    print(classification_report(y_test, y_pred))
    # 혼동 행렬 계산: 실제값 vs 예측값의 교차표
    conf_max = confusion_matrix(y_test, y_pred)
    # 혼동 행렬을 시각화하여 화면에 표시
    ConfusionMatrixDisplay(conf_max).plot()
    # matplotlib 그래프를 화면에 표시
    plt.show()


def run_week1() -> None:
    """Week1의 모든 실습을 순차적으로 실행하는 메인 함수."""
    print("=== Week 1: Iris Dataset Analysis ===")

    # 1. 데이터 로드 및 기본 정보 확인
    print("\n1. 데이터 로드 및 기본 정보:")
    iris = load_iris_data()
    iris_head(iris)

    # 2. 시각화: pandas vs seaborn 산점도 비교
    print("\n2. 산점도 시각화 (pandas vs seaborn):")
    iris_scatter(iris)

    # 3. 시계열 시각화
    print("\n3. Species별 시계열 시각화:")
    iris_series(iris)

    # 4. 머신러닝: 라벨 인코딩 및 Random Forest 분류
    print("\n4. 머신러닝 분석:")
    iris_label_encoder(iris)

    print("\n=== Week 1 Complete! ===")
