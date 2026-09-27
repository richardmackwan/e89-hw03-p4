"""Print the total number of parameters in ImageClassifier."""

from model import ImageClassifier


def main():
    model = ImageClassifier()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total model parameters: {parameter_count:,}")


if __name__ == "__main__":
    main()
