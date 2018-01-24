"""Do linear algebra."""

import numbers
import math


class DimensionError(Exception):
    """Dimensions of matrices don't match or matrix isn't square."""

    pass


class RaggedError(Exception):
    """The rows are not all the same length."""

    pass


class NotInvertibleError(Exception):
    """The matrix is square but not invertible."""

    pass


class DepthError(Exception):
    """The array is not of depth 1 or 2."""

    pass


def depth(array):
    """Return the dimensions of an array, ensuring that it is rectangular."""
    try:
        first = depth(array[0])
        for sub in array:
            if depth(sub) != first:
                raise RaggedError
        return [len(array)] + first
    except TypeError:
        return []
    except IndexError:
        return [0]


class Matrix():
    """This is a matrix."""

    def __init__(self, data):
        """Initialize the matrix."""
        if len(depth(data)) == 2:
            self.data = data
        elif len(depth(data)) == 1:
            self.data = [[item] for item in data]
        else:
            raise DepthError

    def zero(m, n):
        """Initialize a zero matrix with dimension m * n."""
        return Matrix([[0 for j in range(n)] for i in range(m)])

    def fromdiag(diag):
        """Create a square diagonal matrix with a given diagonal."""
        result = Matrix.zero(len(diag), len(diag))
        for i in range(len(diag)):
            result[i, i] = diag[i]
        return result

    def __str__(self):
        """String form."""
        return ''.join([(''.join([(str(round(self[i, j], 3)) + "\t")
                        for j in range(self.n())]) + "\n")
                        for i in range(self.m())])

    __repr__ = __str__

    def repr(self):
        """Implement repr."""
        return "Matrix("+repr(self.data)+")"

    def __add__(self, b):
        """Add."""
        if not self.dim() == b.dim():
            raise DimensionError
        else:
            A = Matrix([[]])
            A.data = [[self[i, j] + b[i, j] for j in range(self.n())]
                      for i in range(self.m())]
            return A

    def __setitem__(self, pos, value):
        """Get an item (duh)."""
        i, j = pos
        self.data[i][j] = value

    def __getitem__(self, pos):
        """Get an item (duh)."""
        i, j = pos
        return self.data[i][j]

    def dim(self):
        """Return the dimensions of a matrix (width x height)."""
        return (len(self.data), len(self.data[0]))

    def m(self):
        """Return m."""
        try:
            return self.dim()[0]
        except IndexError:
            return 0

    def n(self):
        """Return n."""
        try:
            return self.dim()[1]
        except IndexError:
            return 0

    def t(self):
        """Take the transpose of A."""
        result = [[col[i] for col in self.data]
                  for i in range(0, depth(self.data)[1])]
        return Matrix(result)

    def __eq__(self, b):
        """Implement equality for all. Liberty too."""
        for i in range(self.m()):
            for j in range(self.n()):
                if math.fabs(self[i, j] - b[i, j]) > 10**-10:
                    return False
        return True

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

    def cjoin(self, b):
        """Join horizontally: [A B]."""
        return Matrix(self.t().data + b.t().data).t()
        # TODO: Make cjoin more efficient

    def rjoin(self, b):
        """Join vertically."""
        return Matrix(self.data + b.data)

    def cslice(self, j0, j1):
        """Take a horizontal slice from column j0 (incl.) to j1 (excl.)."""
        return Matrix(self.t().data[j0:j1]).t()

    def rslice(self, i0, i1):
        """Take a horizontal slice from row i0 (incl.) to i1 (excl.)."""
        return Matrix(self.data[i0:i1])

    def minor(self, i, j):
        """Return the matrix obtained by removing row i and column j."""
        vdone = Matrix(self.data[0:i] + self.data[i+1:self.m()]).t()
        hdone = Matrix(vdone.data[0:j] + vdone.data[j+1:vdone.m()]).t()
        return hdone

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
            result = Matrix.zero(self.m(), b.n())
            for i in range(self.m()):
                for j in range(b.n()):
                    result[i, j] = sum([(self[i, k] * b[k, j])
                                       for k in range(self.n())])
            return result
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

    def refmrsub(self, i, j):
        """Matrix to (almost) clear column j below row i by subtracting"""
        """copies of row i."""
        e = Matrix.i(self.m())
        for k in range(self.m()):
            if k > i:
                e[k, i] = -self[k, j]/float(self[i, j])
        return e

    def rsub(self, i, j):
        """Clear column j (almost) by subtracting copies of row i."""
        self.rdiv(i, j)
        result = self.mrsub(i, j) * self
        self.data = result.data

    def refrsub(self, i, j):
        """Clear column j below row i (almost)"""
        """by subtracting copies of row i."""
        result = self.refmrsub(i, j) * self
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

    def refredcol(self, i, j):
        """Clear column j below row i (almost) by [exchanging] and subtracting row i."""
        """Return True iff a pivot was created."""
        for k in range(i, self.m()):
            if self[k, j] != 0:
                self.rex(i, k)
                self.refrsub(i, j)
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

    def ref(self):
        """Put in row echelon form."""
        copy = self * 1
        i = 0
        for j in range(copy.n()):
            if copy.refredcol(i, j):
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
        A = self.cjoin(Matrix.identity(self.m()))
        A = A.rref()
        if A.cslice(0, self.m()) != Matrix.identity(self.m()):
            raise NotInvertibleError
        return A.cslice(self.m(), 2*(self.m()))

    def solve(self, b):
        """Solve the system (self) * x = b."""
        A = self.cjoin(b)
        A = A.rref()
        if A.cslice(0, self.m()) == Matrix.identity(self.m()):
            return A.cslice(self.m(), self.m() + 1)
        else:
            raise NotInvertibleError

    def pivotCols(self):
        """Identify pivot columns."""
        copy = self * 1
        i = 0
        pivotCols = []
        for j in range(copy.n()):
            if copy.redcol(i, j):
                i += 1
                pivotCols.append(j)
            if i == self.m():
                return pivotCols
        return pivotCols

    def col(self):
        """Compute the column space."""
        pc = self.pivotCols()
        result = []
        for col in pc:
            result.append(Matrix(self.t().data[col]))
        return VSpace(result, trust=True)

    def indep(self):
        """Return True if the columns of the matrix are independent."""
        return (self.rank() == self.n())

    def allZeros(row):
        """Return True if a row is all zeros."""
        for j in row:
            if j != 0:
                return False
        return True

    def stripZeroRows(self):
        """Remove the zero rows from a matrix."""
        rows = self.data
        result = []
        for row in rows:
            if not Matrix.allZeros(row):
                result.append(row)
        return Matrix(result)

    def null(self):
        """Return the nullspace."""
        pc = self.pivotCols()
        r = self.rref().stripZeroRows().t().data
        free = Matrix.zero(self.rank(), 1)
        for j in range(len(r)):
            if j not in pc:
                bob = Matrix([[i] for i in r[j]])
                free = free.cjoin(bob)
        width = free.n() - 1
        ident = Matrix.identity(width + 1)
        fRow = 0
        iRow = 0
        null = Matrix.zero(len(r), width)
        for i in range(len(r)):
            if i in pc:
                for j in range(width):
                    null[i, j] = -(free[fRow, j+1])
                    fRow += 1
            else:
                for j in range(width):
                    null[i, j] = ident[iRow, j]
        return VSpace([Matrix(vec) for vec in null.t().data], trust=True)

    def isnull(self):
        """Return True iff all the entries are 0."""
        for col in self.data:
            for cell in col:
                if cell != 0:
                    return False
        return True

    def dot(self, b):
        """Return a dot b."""
        return (self.t() * b)[0, 0]

    def orthogonal(self, b):
        """Return True iff orthogonal to b."""
        return self.dot(b) == 0

    def magnitude(self):
        """Return the magnitude."""
        return math.sqrt(self.dot(self))

    ortho = orthogonal
    mag = magnitude

    def sq(self):
        """Test whether the matrix is square."""
        if self.m() == self.n():
            return True
        else:
            return False

    def approxSolve(self, b):
        """Get the approximate solution to Ax==b."""
        return (self.t()*self).inv()*self.t()*b

    def det(self):
        """Calculate the determinant of A."""
        assert self.sq()
        if self.m() == 1:
            return self[0, 0]
        else:
            total = 0
            positive = 1
            for j in range(self.n()):
                total += self[0, j]*self.minor(0, j).det()*positive
                positive = -positive
            return total

    def __pow__(self, n):
        """Raise A to the nth power."""
        assert self.sq()
        total = Matrix.i(self.m())
        for i in range(n):
            total = total * self
        return total

    def trace(self):
        """Return the trace."""
        assert self.sq()
        total = 0
        for i in range(self.m()):
            total = total + self[i, i]
        return total

    def diag(self):
        """Return the diagonal."""
        assert self.sq()
        return [self[i, i] for i in range(self.m())]

    def cofactor(self):
        """Return a cofactor matrix."""
        assert self.sq()
        result = Matrix.zero(self.m(), self.m())
        for i in range(self.m()):
            for j in range(self.m()):
                result[i, j] = self.minor(i, j).det() * (-1) ** (i + j)
        return result

    def pivots(self):
        """Return the pivots."""
        r = self.ref()
        result = []
        for row in r.data:
            for cell in row:
                if cell != 0:
                    result.append(cell)
                    break
        return result

    def posdef(self):
        """Return true if A is positive definite."""
        for item in self.pivots():
            if item <= 0:
                return False
        return True

    def linv(self):
        """Return the left inverse of A."""
        return (self.t() * self).inv() * self.t()

    def rinv(self):
        """Return the right inverse of A."""
        return self.t() * (self * self.t()).inv()

class VSpace():
    """Vector space."""

    def __init__(self, elements, trust=False):
        """Initialize."""
        if trust:
            self.basis = elements
        else:
            A = Matrix([vec.t().data[0] for vec in elements])
            self.basis = A.t().col().basis

    def dim(self):
        """Return the dimension."""
        return len(self.basis)

    def __repr__(self):
        """Input form."""
        return "VSpace(" + self.basis.__repr__() + ")"

    def b(self):
        """Return the basis."""
        return self.basis

    def contains(self, x):
        """Return True iff the vector space contains x."""
        return (self.dim() == VSpace(self.basis + [x]).dim())
