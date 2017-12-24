"""Test the linalg module."""
from linalg import *
import unittest


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
