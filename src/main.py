from .week1 import iris_head, iris_scatter, load_iris_data


def main() -> None:
    # Load and display iris data from week1
    print("\nLoading iris data from week1:")  # noqa: T201
    iris = load_iris_data()
    iris_head(iris)
    iris_scatter(iris)


if __name__ == "__main__":
    main()
