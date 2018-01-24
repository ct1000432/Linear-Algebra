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

    def test_join_cslice(self):
        """Test rjoin, cjoin and cslice."""
        self.assertEqual(self.A.cjoin(self.C),
                         Matrix([[7, 21, 3, 3, 4, 0],
                                 [-47, 14, 3, -1, 0, 2]]))
        self.assertEqual(self.A.cjoin(self.C).cslice(0, 3), self.A)
        self.assertEqual(self.A.rjoin(self.C),
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
                self.assertTrue(u.cjoin(v).indep() or u == v)

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

    def test_approxSolve(self):
        """Test approximate solve."""
        self.assertEqual(self.B.t().approxSolve(Matrix([1, 1, 2, 3])),
                         Matrix([-0.0648653138797121,
                                 0.0126069718907576,
                                 0.0627138888661882]))

    def test_ref(self):
        """Test row-echelon form."""
        self.assertEqual(self.B.ref(),
                         Matrix([
                                    [55.0, -21.0, 13.0, -27.0],
                                    [
                                        0.0,
                                        65.12727272727273,
                                        6.254545454545454,
                                        30.163636363636364
                                    ],
                                    [
                                        0.0,
                                        0.0,
                                        37.382467895030715,
                                        16.82998324958124
                                    ]
                                ]))

    def test_minor(self):
        """Test minor()."""
        self.assertEqual(self.B.minor(1, 2),
                         Matrix([
                                 [55, -21, -27],
                                 [56, -7, -4]]))

    def test_det(self):
        """Test determinant."""
        self.assertEqual(self.D.det(), 0)
        self.assertEqual(self.E.det(), -3)

    def test_pow(self):
        """Test power."""
        self.assertEqual(self.E ** 5, Matrix([[-5,  -78,   83],
                                              [-452, -753,  239],
                                              [405,  561,  -83]]))
        self.assertEqual(self.D ** 5, Matrix([[-59, -132,   73],
                                              [-542, -879,  337],
                                              [483,  747, -264]]))

    def test_trace(self):
        """Test trace."""
        self.assertEqual(self.D.trace(), -2)

    def test_diag(self):
        """Test diagonal."""
        self.assertEqual(self.D.diag(), [1, -3, 0])
        self.assertEqual(Matrix.fromdiag([3, 7, 1]), Matrix([
            [3, 0, 0],
            [0, 7, 0],
            [0, 0, 1]
        ]))

    def test_cofactor(self):
        """Test cofactor."""
        self.assertEqual(self.E.cofactor(),
                         Matrix([[-6, 5, 3], [3, -2, -3], [3, -3, -3]]))

    def test_pivots(self):
        """Test pivots."""
        self.assertEqual(self.D.pivots(), [1, -3])
        self.assertEqual(self.E.pivots(), [1, -3, 1])

    def test_posdef(self):
        """Test positive definite."""
        self.assertTrue(self.A.posdef())
        self.assertTrue(self.B.posdef())
        self.assertFalse(self.D.posdef())

    def test_linv_rinv(self):
        """Test left inverse and right inverse."""
        self.assertEqual(self.B * self.B.rinv(), Matrix.i(3))
        Q = self.A.t()
        self.assertEqual(Q.linv() * Q, Matrix.i(2))


if __name__ == '__main__':
    unittest.main()
