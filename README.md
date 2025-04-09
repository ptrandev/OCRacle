# OCRacle

Developer Names: Phillip Tran

Date of project start: January 13, 2025

This project is an optical character recognition tool for detecting Latin alphabet characters.

The folders and files for this project are as follows:

docs - Documentation for the project
refs - Reference material used for the project, including papers
src - Source code
test - Test cases

## Installation

For installation instructions, reference the INSTALL.md file in the root directory of the project.

## Usage

### Training the Model

To train the model, you can use the following command:
```bash
python src/train.py
```

This will train the model and save it to `src/model.keras`.

### Evaluating the Model

To evaluate the model's performance, you can use the following command:
```bash
python src/evaluate.py
```

This will produce a report on the model's performance, including accuracy and loss metrics:

```

```

### Using the Model

### Running the Test Suite

To run the test suite, you can use the following command:
```bash
pytest test/test.py
```

This will produce a test report in the terminal:

```

```

This will run all the test cases in the `test/test.py` file. You can also run individual test cases by specifying their names.
```bash
pytest test/test.py::test_case_name
```

If you are running the entire test suite, please note that one of the tests is
to train the model. This may take a few minutes to run, depending on your
system.