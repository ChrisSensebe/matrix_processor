import functools
import math


class MatrixErrors:
    INVALID_MATRIX = "Invalid matrix"
    INVALID_OPERATION = "The operation cannot be performed."
    NO_INVERSE_MATRIX = "This matrix doesn't have an inverse."


class MatrixProcessorState:

    EXIT = 0
    ADD_MATRICES = 1
    MULTIPLY_SCALAR = 2
    MULTIPLY_OTHER = 3
    TRANSPOSE_MATRIX = 4
    CALC_DETERMINANT = 5
    INVERSE = 6
    MAIN_MENU = 7

    def __init__(self):
        self.__state = MatrixProcessorState.MAIN_MENU

    def set_state(self, state):
        """
        set processor_state state
        :param state: new state
        """
        self.__state = state

    def get_state(self):
        """
        get processor_state state
        :return: current state
        """
        return self.__state


class MatrixTranspositions:

    MAIN_DIAGONAL = 1
    SIDE_DIAGONAL = 2
    VERTICAL_LINE = 3
    HORIZONTAL_LINE = 4


class MatrixProcessor:

    def __init__(self):
        self.processorState = MatrixProcessorState()

    def run(self):
        """
        main loop
        """
        while self.processorState.get_state() != MatrixProcessorState.EXIT:
            try:
                if self.processorState.get_state() == MatrixProcessorState.ADD_MATRICES:
                    self.__add_matrices()
                elif self.processorState.get_state() == MatrixProcessorState.MULTIPLY_SCALAR:
                    self.__multiply_scalar()
                elif self.processorState.get_state() == MatrixProcessorState.MULTIPLY_OTHER:
                    self.__multiply_other()
                elif self.processorState.get_state() == MatrixProcessorState.TRANSPOSE_MATRIX:
                    self.transpose_matrix()
                elif self.processorState.get_state() == MatrixProcessorState.CALC_DETERMINANT:
                    self.calc_determinant()
                elif self.processorState.get_state() == MatrixProcessorState.INVERSE:
                    self.inverse_matrix()
                else:
                    self.__set_state_main_menu()
            except Exception as error:
                print(f"{error}\n")
                self.processorState.set_state(MatrixProcessorState.MAIN_MENU)

    def __add_matrices(self):
        """
        input two matrices and add them
        """
        first_matrix = MatrixConsoleAdapter.input_matrix(1)
        second_matrix = MatrixConsoleAdapter.input_matrix(2)
        result_matrix = first_matrix.add(second_matrix)
        MatrixConsoleAdapter.print_result(result_matrix)
        self.processorState.set_state(MatrixProcessorState.MAIN_MENU)

    def __multiply_scalar(self):
        """
        input matrix and scalar and multiply them
        """
        matrix = MatrixConsoleAdapter.input_matrix(-1)
        scalar = MatrixConsoleAdapter.input_constant()
        result_matrix = matrix.multiply_scalar(scalar)
        MatrixConsoleAdapter.print_result(result_matrix)
        self.processorState.set_state(MatrixProcessorState.MAIN_MENU)

    def __multiply_other(self):
        """
        input two matrices and multiply them
        """
        first_matrix = MatrixConsoleAdapter.input_matrix(1)
        second_matrix = MatrixConsoleAdapter.input_matrix(2)
        result_matrix = first_matrix.multiply_other(second_matrix)
        MatrixConsoleAdapter.print_result(result_matrix)
        self.processorState.set_state(MatrixProcessorState.MAIN_MENU)

    def transpose_matrix(self):
        """
        choose a transposition
        input a matrix and transpose it
        """
        transposition = MatrixConsoleAdapter.choose_transposition()
        matrix = MatrixConsoleAdapter.input_matrix(-1)
        result_matrix = None
        if transposition == MatrixTranspositions.MAIN_DIAGONAL:
            result_matrix = matrix.transpose()
        elif transposition == MatrixTranspositions.SIDE_DIAGONAL:
            result_matrix = matrix.reverse_transpose()
        elif transposition == MatrixTranspositions.VERTICAL_LINE:
            result_matrix = matrix.vertical_transpose()
        elif transposition == MatrixTranspositions.HORIZONTAL_LINE:
            result_matrix = matrix.horizontal_transpose()
        MatrixConsoleAdapter.print_result(result_matrix)
        self.processorState.set_state(MatrixProcessorState.MAIN_MENU)

    def calc_determinant(self):
        """
        input a matrix and compute it determinant
        """
        matrix = MatrixConsoleAdapter.input_matrix(-1)
        MatrixConsoleAdapter.print_result(matrix.calc_determinant())
        self.processorState.set_state(MatrixProcessorState.MAIN_MENU)

    def inverse_matrix(self):
        """
        input a matrix and compute it inverse
        """
        matrix = MatrixConsoleAdapter.input_matrix(-1)
        MatrixConsoleAdapter.print_result(matrix.inverse())
        self.processorState.set_state(MatrixProcessorState.MAIN_MENU)

    def __set_state_main_menu(self):
        """
        set processot state to main menu
        """
        self.processorState.set_state(MatrixConsoleAdapter.main_menu())


class Matrix:

    def __init__(self, table):
        is_valid = len(set([len(row) for row in table])) <= 1
        if not is_valid:
            raise Exception(MatrixErrors.INVALID_MATRIX)
        self.__table = tuple([tuple(row) for row in table])
        self.__dimensions = (len(table), len(table[0]))

    def __str__(self):
        return functools.reduce(
            lambda prev, curr: prev + " ".join([str(entry) for entry in curr]) + "\n",
            self.__table,
            "").rstrip("\n")

    def get_dimensions(self):
        """
        matrix dimensions
        :return: (rows, columns)
        """
        return self.__dimensions

    def get_column(self, index_column):
        """
        matrix column at index
        :param index_column:
        :return: column
        """
        return [row[index_column] for row in self.__table]

    def get_row(self, index_row):
        """
        matrix row at index
        :param index_row:
        :return: row
        """
        return list(self.__table[index_row])

    def add(self, other):
        """
        add matrix to self
        :param other: matrix to add
        :return: result matrix
        """
        if other.__dimensions[0] != self.__dimensions[0] or other.__dimensions[1] != self.__dimensions[1]:
            raise Exception(MatrixErrors.INVALID_OPERATION)
        new_table = []
        for line in range(self.__dimensions[0]):
            if len(self.__table[line]) != len(other.__table[line]):
                raise Exception(MatrixErrors.INVALID_OPERATION)
            new_line = []
            for i in range(self.__dimensions[1]):
                new_line.append(self.__table[line][i] + other.__table[line][i])
            new_table.append(new_line)
        return Matrix(new_table)

    def multiply_scalar(self, scalar):
        """
        multiply self by scalar
        :param scalar: scalar
        :return: result matrix
        """
        new_table = []
        for line in range(self.__dimensions[0]):
            new_table.append([scalar * number for number in self.__table[line]])
        return Matrix(new_table)

    def multiply_other(self, other):
        """
        multiply by other matrix
        :param other: other matrix
        :return: result matrix
        """
        self_rows = self.get_dimensions()[0]
        self_columns = self.get_dimensions()[1]
        other_rows = other.get_dimensions()[0]
        other_columns = other.get_dimensions()[1]
        if self_columns != other_rows:
            raise Exception(MatrixErrors.INVALID_OPERATION)
        new_table = []
        for i in range(self_rows):
            new_row = []
            row = self.get_row(i)
            for j in range(other_columns):
                other_column = other.get_column(j)
                new_row_elem = sum([row[index] * other_column[index] for index in range(self_columns)])
                new_row.append(new_row_elem)
            new_table.append(new_row)
        return Matrix(new_table)

    def transpose(self):
        """
        transpose matrix
        :return: transposed matrix
        """
        return Matrix([self.get_column(index) for index in range(self.get_dimensions()[0])])

    def reverse_transpose(self):
        """
        transpose matrix along reverse diagonal
        :return: transposed matrix
        """
        first_step = [self.get_column(index) for index in reversed(range(self.get_dimensions()[0]))]
        return Matrix([row[::-1] for row in first_step])

    def vertical_transpose(self):
        """
        transpose matrix along vertical axis
        :return: transposed matrix
        """
        return Matrix([row[::-1] for row in self.__table])

    def horizontal_transpose(self):
        """
        transpose matrix along horizontal axis
        :return: transposed matrix
        """
        return Matrix(self.__table[::-1])

    def calc_determinant(self):
        """
        compute matrix determinant
        :return: the determinant
        """
        if self.__dimensions[0] != self.__dimensions[1]:
            raise Exception(MatrixErrors.INVALID_OPERATION)
        if self.__dimensions[0] == 1:
            return self.__table[0][0]
        if self.__dimensions[0] == 2:
            return self.__table[0][0] * self.__table[1][1] - self.__table[0][1] * self.__table[1][0]
        else:
            zero_count = [row.count(0) for row in self.__table]
            excluded_row = zero_count.index(max(zero_count))
            cofactors = []
            matrix_size = range(len(self.__table[0]))
            for i in matrix_size:
                table = []
                for j in matrix_size:
                    if j != excluded_row:
                        row = self.get_row(j)
                        del row[i]
                        table.append(row)
                cofactor = self.__table[excluded_row][i]
                minor_det = (1 if i % 2 == 0 else -1) * cofactor * Matrix(table).calc_determinant() if cofactor != 0 else 0
                cofactors.append(minor_det)
            return sum(cofactors)

    def inverse(self):
        """
        compute inverse matrix
        :return: the inverse matrix
        """
        determinant = self.calc_determinant()
        if determinant == 0:
            raise Exception(MatrixErrors.NO_INVERSE_MATRIX)
        return self.__cofactor_matrix() \
            .transpose() \
            .multiply_scalar(1 / self.calc_determinant())

    def __cofactor_matrix(self):
        if self.__dimensions[0] != self.__dimensions[1]:
            raise Exception(MatrixErrors.INVALID_OPERATION)
        if self.__dimensions[0] == 1:
            return Matrix([[1]])
        if self.__dimensions[0] == 2:
            table = list(self.__table)
            table[0][1] *= -1
            table[1][0] *= -1
            return Matrix(table).reverse_transpose()
        else:
            cofactors = []
            matrix_size = len(self.__table)
            for i in range(matrix_size):
                rows = list(self.__table)
                del rows[i]
                row_cofactors = []
                for j in range(matrix_size):
                    minor = []
                    for row in rows:
                        new_row = list(row)
                        del new_row[j]
                        minor.append(new_row)
                    factor = -1 if (i % 2 == 0 and j % 2 != 0) or (i % 2 != 0 and j % 2 == 0) else 1
                    row_cofactors.append(factor * Matrix(minor).calc_determinant())
                cofactors.append(row_cofactors)
        return Matrix(cofactors)


class MatrixConsoleAdapter:

    RESULT_PROMPT = "The result is:"

    @staticmethod
    def main_menu():
        """
        print main menu
        :return: user choice
        """
        print("1. Add matrices")
        print("2. Multiply matrix by a constant")
        print("3. Multiply matrices")
        print("4. Transpose matrix")
        print("5. Calculate a determinant")
        print("6. Inverse matrix")
        print("0. Exit")
        choice = None
        while choice not in range(7):
            choice = int(input("Your choice: "))
        return choice

    @staticmethod
    def input_matrix(matrix_number):
        """
        input a matrix
        :param matrix_number: matrix number, -1 if no number
        :return: new matrix
        """
        first_prompt, second_prompt = MatrixConsoleAdapter.__get_prompt(matrix_number)
        dimensions = MatrixConsoleAdapter.__input_dimensions(first_prompt)
        return Matrix(MatrixConsoleAdapter.__input_table(dimensions, second_prompt))

    @staticmethod
    def input_constant():
        """
        input a constant
        :return: user input
        """
        return MatrixConsoleAdapter.__get_input_number(input("Enter constant: "))

    @staticmethod
    def __input_dimensions(prompt):
        """
        input matrix dimensions
        :param prompt: input prompt
        :return: matrix dimensions  (list)
        """
        return [MatrixConsoleAdapter.__get_input_number(dimension) for dimension in input(prompt).split(" ")]

    @staticmethod
    def __input_table(dimensions, prompt):
        """
        input matrix representation
        :param dimensions: matrix dimensions (list)
        :param prompt: input prompt
        :return: matrix representation (list)
        """
        print(prompt)
        table_input = [input().split(" ") for line in range(dimensions[0])]
        table = []
        for line in table_input:
            table_line = [MatrixConsoleAdapter.__get_input_number(number) for number in line]
            table.append(table_line)
        return table

    @staticmethod
    def __get_prompt(matrix_number):
        """
        get prompt for matrix input
        :param matrix_number: -1 for no number
        :return: prompt
        """
        matrix_name = ""
        if matrix_number == 1:
            matrix_name = "first "
        elif matrix_number == 2:
            matrix_name = "second "
        return f"Enter size of {matrix_name}matrix: ", f"Enter {matrix_name}Matrix:"

    @staticmethod
    def __get_input_number(number):
        """
        input a float or int from console
        :param number: number (string)
        :return: number (int or float)
        """
        try:
            return int(number)
        except ValueError:
            return float(number)

    @staticmethod
    def choose_transposition():
        """
        menu to choose matrix transposition
        :return: user choice
        """
        transposition = None
        while transposition not in range(1, 5):
            print("1. Main diagonal")
            print("2. Side diagonal")
            print("3. Vertical line")
            print("4. Horizontal line")
            transposition = int(input('Your choice: '))
        return transposition

    @staticmethod
    def print_result(result_matrix):
        """
        print matrix to console
        :param result_matrix:
        """
        print(MatrixConsoleAdapter.RESULT_PROMPT)
        print(result_matrix)
        print("")



processor = MatrixProcessor()
processor.run()
