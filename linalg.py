"""Do linear algebra."""


class DimensionError(Exception):
    """The dimensions of the matrices don't match."""

    pass


class NotRectangularError(Exception):
    """The rows are not all the same length."""

    pass


class Vector():
    """Be awesome."""

    def __init__(self, array):
        """Initialize a new vector."""
        self.data = array

    pass


class Matrix():
    """This is a matrix."""

    def __init__(self, array):
        """Initialize the thingy."""
        n = len(array[0])
        for row in array:
            if len(row) != n:
                raise NotRectangularError
        self.data = array

    def __add__(self, b):
        """Be magic."""
        if not self.dim() == b.dim():
            raise DimensionError
        else:
            return [[self[i][j] + b[i][j] for j in self.dim()[1]]
                    for i in self.dim()[0]]

    def __getitem__(self, index):
        return self.data[index]

    def dim(self):
        """Return the dimensions of a matrix."""
        return (len(self), len(self[0]))

    def t(self):
        """Take the transpose of A."""
        return [[row[i] for row in self] for i in range(0, len(self[0]))]

    def __mul__(self, b):
        """Multiply two matrices or a matrix and a scalar."""
        if type(b) is int:
            return[[a * b for a in row] for row in self]
        # elif type(b) is Matrix:
            # if not dim(a)[0] == dim(b)[1]:
            #    raise DimensionError
            #    for row in self:
            #        for row in t(b):    # TODO: finish
            #            print(1+1)
        else:
            raise TypeError
