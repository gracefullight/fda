from .week1 import iris_head, iris_label_encoder, load_iris_data


def main() -> None:
    # Load and display iris data from week1
    print("\nLoading iris data from week1:")  # noqa: T201
    iris = load_iris_data()
    iris_head(iris)
    # iris_scatter(iris)
    # iris_series(iris)
    iris_label_encoder(iris)


if __name__ == "__main__":
    main()
