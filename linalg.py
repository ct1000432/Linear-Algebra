"""Do linear algebra."""

import numbers


class DimensionError(Exception):
    """Dimensions of matrices don't match or matrix isn't square."""

    pass


class NotRectangularError(Exception):
    """The rows are not all the same length."""

    pass


class NotInvertibleError(Exception):
    """The matrix is square but not invertible."""

    pass


class Matrix():
    """This is a matrix."""

    def __init__(self, array):
        """Initialize the matrix."""
        m = len(array[0])
        for row in array:
            if len(row) != m:
                raise NotRectangularError
        self.data = array
        self.data = [[col[i] for col in self.data]
                     for i in range(0, len(array[0]))]

    def zero(m, n):
        """Initialize a zero matrix with dimension m * n."""
        return Matrix([[0 for j in range(n)] for i in range(m)])

    def __str__(self):
        """String form."""
        return ''.join([(''.join([(str(self[i, j]) + "\t")
                        for j in range(self.n())]) + "\n")
                        for i in range(self.m())])

    def __repr__(self):
        """Implement repr."""
        return "Matrix("+repr(self.t().data)+")"

    def __add__(self, b):
        """Add."""
        if not self.dim() == b.dim():
            raise DimensionError
        else:
            A = Matrix([[]])
            A.data = [[self[i, j] + b[i, j] for j in range(self.n())]
                      for i in range(self.m())]
            return A.t()

    def __setitem__(self, pos, value):
        """Get an item (duh)."""
        i, j = pos
        self.data[j][i] = value

    def __getitem__(self, pos):
        """Get an item (duh)."""
        i, j = pos
        return self.data[j][i]

    def dim(self):
        """Return the dimensions of a matrix (width x height)."""
        return (len(self.data[0]), len(self.data))

    def m(self):
        """Return m."""
        return self.dim()[0]

    def n(self):
        """Return n."""
        return self.dim()[1]

    def t(self):
        """Take the transpose of A."""
        return Matrix(self.data)

    def __eq__(self, b):
        """Implement equality for all. Liberty too."""
        return (self.data == b.data)

    def __neg__(self):
        """Implement negation."""
        return self * -1

    def __sub__(self, b):
        """Implement subtraction."""
        return self + (-b)

    def __truediv__(self, b):
        """Implement scalar division."""
        if isinstance(b, numbers.Number):
            return self * (1.0/b)
        else:
            raise TypeError

    def hjoin(self, b):
        """Join horizontally: [A B]."""
        return Matrix(self.data + b.data).t()

    def vjoin(self, b):
        """Join vertically."""
        return Matrix(self.t().data + b.t().data)

    def vslice(self, j0, j1):
        """Take a vertical slice from column j0 (incl.) to j1 (excl.)."""
        return Matrix(self.data[j0:j1]).t()

    def symmetric(self):
        """Test for symmetry."""
        return (self == self.t())

    symm = symmetric

    def identity(m):
        """Return the m * m identity matrix."""
        z = Matrix.zero(m, m)
        for i in range(m):
            z[i, i] = 1
        return z

    i = identity

    def __mul__(self, b):
        """Multiply two matrices or a matrix and a scalar."""
        if isinstance(b, numbers.Number):
            return Matrix([[self[i, j] * b for j in range(self.n())]
                           for i in range(self.m())])
        elif type(b) is Matrix:
            if not self.n() == b.m():
                raise DimensionError
            else:
                data = Matrix.zero(self.m(), b.n()).t().data
                for i in range(self.m()):
                    for j in range(b.n()):
                        data[i][j] = sum([(self[i, k] * b[k, j])
                                          for k in range(self.n())])
                return Matrix(data)
        else:
            print(type(b))
            raise TypeError

    __rmul__ = __mul__

    def mrex(self, i1, i2):
        """Matrix to exchange rows i1 and i2."""
        e = Matrix.i(self.m())
        e[i1, i1] = 0
        e[i2, i2] = 0
        e[i1, i2] = 1
        e[i2, i1] = 1
        return e

    def rex(self, i1, i2):
        """Exchange rows i1 and i2."""
        result = self.mrex(i1, i2) * self
        self.data = result.data

    def mrdiv(self, i, div):
        """Matrix to divide row i by div."""
        e = Matrix.i(self.m())
        e[i, i] = 1.0/div
        return e

    def rdiv(self, i, j):
        """Divide row i by item in (i, j)."""
        result = self.mrdiv(i, self[i, j]) * self
        self.data = result.data

    def mrsub(self, i, j):
        """Matrix to (almost) clear column j by subtracting copies of row i."""
        e = Matrix.i(self.m())
        for k in range(self.m()):
            if k != i:
                e[k, i] = -self[k, j]
        return e

    def rsub(self, i, j):
        """Clear column j (almost) by subtracting copies of row i."""
        self.rdiv(i, j)
        result = self.mrsub(i, j) * self
        self.data = result.data

    def redcol(self, i, j):
        """Clear column j (almost) by [exchanging] and subtracting row i."""
        """Return True iff a pivot was created."""
        for k in range(i, self.m()):
            if self[k, j] != 0:
                self.rex(i, k)
                self.rsub(i, j)
                return True
        return False

    def rref(self):
        """Put in reduced row echelon form."""
        copy = self * 1
        i = 0
        for j in range(copy.n()):
            if copy.redcol(i, j):
                i += 1
            if i == self.m():
                return copy
        return copy

    def rank(self):
        """Get the rank."""
        copy = self * 1
        i = 0
        for j in range(copy.n()):
            if copy.redcol(i, j):
                i += 1
            if i == self.m():
                return i
        return i

    def inv(self):
        """Invert the matrix."""
        if self.m() != self.n():
            raise DimensionError
        A = self.hjoin(Matrix.identity(self.m()))
        A = A.rref()
        if A.vslice(0, self.m()) != Matrix.identity(self.m()):
            raise NotInvertibleError
        return A.vslice(self.m(), 2*(self.m()))

    def solve(self, b):
        """Solve the system (self) * x = b."""
        if type(b) is list:
            B = Matrix([b]).t()
        elif type(b) is Matrix:
            B = b
        else:
            raise TypeError
        A = self.hjoin(B)
        if A.vslice(0, self.m()) == Matrix.identity(self.m()):
            return A.rref().vslice(self.m(), self.m() + 1)
        else:
            raise NotInvertibleError
