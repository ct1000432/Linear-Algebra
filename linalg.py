"""Do linear algebra."""

import numbers
import math
import unittest

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
        self.data = [[col[i] for col in self.data]
                     for i in range(0, depth(self.data)[1])]

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
        return Matrix(self.data)

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

    def hjoin(self, b):
        """Join horizontally: [A B]."""
        if type(b) is list:
            return self.hjoin(Matrix([b]).t())
        return Matrix(self.data + b.data).t()

    def vjoin(self, b):
        """Join vertically."""
        return Matrix(self.t().data + b.t().data)

    def hslice(self, j0, j1):
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
        if A.hslice(0, self.m()) != Matrix.identity(self.m()):
            raise NotInvertibleError
        return A.hslice(self.m(), 2*(self.m()))

    def solve(self, b):
        """Solve the system (self) * x = b."""
        A = self.hjoin(b)
        A = A.rref()
        if A.hslice(0, self.m()) == Matrix.identity(self.m()):
            return A.hslice(self.m(), self.m() + 1)
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
            result.append(Matrix(self.data[col]))
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
        rows = self.t().data
        result = []
        for row in rows:
            if not Matrix.allZeros(row):
                result.append(row)
        return Matrix(result)

    def null(self):
        """Return the nullspace."""
        pc = self.pivotCols()
        r = self.rref().stripZeroRows().data
        free = Matrix.zero(self.rank(), 1)
        for j in range(len(r)):
            if j not in pc:
                free = free.hjoin(r[j])
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
        return VSpace([Matrix(vec) for vec in null.data], trust=True)

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


class VSpace():
    """Vector space."""

    def __init__(self, elements, trust=False):
        """Initialize."""
        if trust:
            self.basis = elements
        else:
            A = Matrix([vec.data[0] for vec in elements])
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


class ArithTest(unittest.TestCase):
    """Test arithmetic."""

    def setUp(self):
        """Set-up."""
        self.A = Matrix([[7, 21, 3], [-47, 14, 3]])
        self.B = Matrix([[55, -21, 13, -27],
                         [37, 51, 15, 12],
                         [56, -7, 52, -4]])
        self.C = Matrix([[3, 4, 0], [-1, 0, 2]])
        self.D = Matrix([[1, 0, 1], [-2, -3, 1], [3, 3, 0]])
        self.E = Matrix([[1, 0, 1], [-2, -3, 1], [3, 3, 1]])
        self.x = Matrix([1, 2, 3])
        self.sum = Matrix([[10, 25, 3], [-48, 14, 5]])
        self.alsoA = Matrix([[7, 21, 3], [-47, 14, 3]])
        self.product = Matrix([[1330, 903, 562, 51],
                               [-1899, 1680, -245, 1425]])

    def test_eq(self):
        """Test equality."""
        self.assertEqual(self.A, self.alsoA)

    def test_matrix_mult(self):
        """Test matrix multiplication."""
        self.assertEqual(self.A*self.B, self.product)
        self.assertRaises(DimensionError, self.A.__mul__, self.C)

    def test_scalar_mult_div(self):
        """Test scalar multiplication and division."""
        self.assertEqual(self.x * 2, Matrix([2, 4, 6]))
        self.assertEqual(-3 * self.x, Matrix([-3, -6, -9]))
        self.assertEqual(self.x / 2, Matrix([.5, 1, 1.5]))

    def test_matrix_add_sub_neg(self):
        """Test matrix addition, negation and subtraction."""
        self.assertEqual(self.A + self.C, self.sum)
        self.assertRaises(DimensionError, self.A.__add__, self.B)
        self.assertEqual(self.sum - self.A, self.C)

    def test_rref(self):
        """Test RREF."""
        self.assertEqual(self.A.rref(),
                         Matrix([[1.0, 0.0, -0.01935483870967747],
                                 [0.0, 1.0, 0.14930875576036867]]))
        self.assertEqual(self.A.rank(), 2)
        self.assertEqual(self.D.rank(), 2)

    def test_null(self):
        """Test nullspace."""
        self.assertTrue(self.D.null().contains(Matrix([-1, 1, 1])))
        self.assertTrue(self.B.t().null().b() == [])

    def test_col(self):
        """Test column space."""
        self.assertTrue(self.D.col().contains(Matrix([0, 0, 0])))
        self.assertTrue(self.D.col().contains(Matrix([2, -4, 6])))
        self.assertFalse(self.D.col().contains(Matrix([2, -1, 0])))

    def test_t(self):
        """Test transpose."""
        self.assertEqual(self.A.t(), Matrix([[7, -47], [21, 14], [3, 3]]))

    def test_inv_id(self):
        """Test inverse and identity."""
        self.assertEqual(self.E.inv()*self.E, Matrix.i(3))
        self.assertRaises(DimensionError, self.B.inv)
        self.assertRaises(NotInvertibleError, self.D.inv)

    def test_join_hslice(self):
        """Test vjoin, hjoin and hslice."""
        self.assertEqual(self.A.hjoin(self.C),
                         Matrix([[7, 21, 3, 3, 4, 0],
                                 [-47, 14, 3, -1, 0, 2]]))
        self.assertEqual(self.A.hjoin(self.C).hslice(0, 3), self.A)
        self.assertEqual(self.A.vjoin(self.C),
                         Matrix([[7, 21, 3],
                                 [-47, 14, 3],
                                 [3, 4, 0],
                                 [-1, 0, 2]]))

    def test_symm(self):
        """Test symmetry."""
        self.assertTrue((self.A.t()*self.A).symm())
        self.assertFalse((self.A.t()*self.C).symm())

    def test_get_set(self):
        """Test getting and setting items."""
        changed = self.A
        changed[1, 1] = -5
        self.assertEqual(self.A[1, 1], -5)
        self.assertEqual(self.A[1, 2], 3)

    def test_zero(self):
        """Test zero."""
        self.assertEqual(Matrix.zero(2, 3), Matrix([[0, 0, 0], [0, 0, 0]]))

    def test_solve(self):
        """Test solve."""
        self.assertEqual(self.E.solve(Matrix([1, 2, 3])),
                         Matrix([-3.0, 2.666666666666667, 4.0]))

    def test_b_indep(self):
        """Test basis and independence."""
        S = self.B.t().col()
        for u in S.b():
            self.assertTrue(S.contains(u))
            for v in S.b():
                self.assertTrue(u.hjoin(v).indep() or u == v)

    def test_dim(self):
        """Test dimension of a vector space."""
        S = self.B.t().col()
        self.assertEqual(S.dim(), 3)

    def test_ortho(self):
        """Test orthogonality."""
        A = Matrix([1, 2, 3])
        B = Matrix([1, 1, -1])
        self.assertTrue(A.ortho(B))
        self.assertFalse(A.ortho(A))
