# Linear Algebra!
Behold! Herein lieth evidence that I have become a competent sorcerer of linear algebra.

## Goals
See the [manifesto thing](https://docs.google.com/document/d/1Xo6KA83j4WAhBD5HMyx-iB5lSi7Xr8Y1wnFqm6Dzb_8/edit?usp=sharing) that I wrote at the beginning of the semester.

## Notes
I've included scans of my notes in the `notes` folder, which I think provide reasonable evidence that I've actually been learning linear algebra, because producing thirty pages of notes (including funny quotes from the professor) without watching the lectures would be harder than just watching the lectures.

There's also a nicely typed-up study guide I made for my Unit 1 exam, which I didn't do for the other two exams because I felt I reviewed those well enough by working on my program.

## Exams
I've taken three exams, written by my ~~master wizard~~ advisor, on the first two of which I've done well and on the last of which I imagine I've done well though I haven't gotten a grade back yet.

## Program
I've compiled a program throughout the semester, which encapsulates most of my linear algebra learnings. Because the course was more about theory than about applications, because (I not being an experienced developer and having only four months) a program for an application would probably not really be very useful, and because I think it demonstrates my understanding much better, I decided to write a program to perform the algorithms we learned in class rather than to apply them. (I left this open in my plan.) My Python skills, unfortunately, are not extensive enough to implement Gram-Schmidt orthogonalization or calculate eigenvalues, so eigenvalues and Gram-Schmidt are not dealt with in the program.

If you have a computer with Python 3 installed, great: download `linalg.py` from the `program` folder. Otherwise, you can use the program online, using [repl.it](https://repl.it/@draziwfozo/Linalg); make sure to hit the "run" button to load the program before entering any commands.

### Python Basics
The dark blue area on repl.it with the pixelated `>` prompt is called the *shell*. When you type commands in Python at the shell and then hit the enter key, they get executed and the shell prints the result, if any, below.

The shell will evaluate arithmetical expressions for you. Type
``` python
 4 + 7
```
after the `>` prompt (spaces are optional) and then press enter. Unless math has suddenly stopped working, it should return `11`. You can use `+` for addition, `-` for subtraction, `*` for multiplication, `/` for division, and `(` and `)` whenever necessary for grouping. Exponentiation is written `**`.

You can assign values to *variables* and use those variables in expressions. For example, type
``` python
x = 5
```
and press enter. Then, type
```python
(x * 3) + 2
```
which should return `8`.

Note that `=` *does not* test if two things are equal. The Python equivalent of the equals sign in math is `==`. For example, typing
``` python
2 - x == -3
```
should return `True`.

### Creating and Displaying Matrices
Typing
```python
A = Matrix([[7, 21, 3], [-47, 14, 3]])
```
assigns the matrix
```
  7  21   3
-47  14   3
```
to the variable `A`.  (You can copy and paste instead of typing.) You can use more or fewer rows and columns, or whatever numbers you want.
```python
B = Matrix([[55, -21, 13, -27],
            [37, 51, 15, 12],
            [56, -7, 52, -4]])
```
will assign a larger matrix to `B`. (Notice that if you put a line break in the middle of an unfinished statement, the prompt will show `..` and let you continue it.)
Finally, run
``` python
C = Matrix([[3, 4, 0], [-1, 0, 2]])
```
``` python
D = Matrix([[1, 0, 1], [-2, -3, 1], [3, 3, 0]])
```
``` python
E = Matrix([[1, 0, 1], [-2, -3, 1], [3, 3, 1]])
```
to assign three more matrices to `C`, `D`, and `E`. Alternately, you can use other matrices, but make sure that `A` and `C` are *m* by *n* and `B` is *n* by *p*, where *m < n < p*; `D` is square and singular; and `E` is square and invertible so that the examples later make sense. Now we have all the matrices we'll need to demonstrate linear algebra!

As a shortcut, you can enter
```python
Matrix([1, 2, 3, 5])
```
instead of
```python
Matrix([[1], [2], [3], [5]])
```
for column vectors.

To view a matrix that is in a variable, type the variable's name at the prompt. For example, typing
```python
B
```
should yield an output of
```
55	-21	13	-27
37	51	15	12
56	-7	52	-4
```
(the data in the matrix `B`.)

### Matrix Arithmetic
Addition, subtraction, and multiplication are defined for matrices.

Matrices can only be added and subtracted if they have the same dimensions:
```python
A + C
```
should yield a sum, but
```python
A + B
```
will give an error message (ending in the type of error, namely `DimensionError`). The same holds for subtraction.

Matrices can be multiplied only if the *width* of the first equals the *height* of the second. Matrix multiplication is not commutative, meaning that `A * B == B * A` is not generally true (and in fact often isn't defined). Running
```python
A * B
```
should give an answer, but
```python
B * A
```
should raise `DimensionError`.

You can also multiply a matrix by a regular number (called a *scalar* in the context of linear algebra). For instance,
```python
A * 2
```
will give the same answer as
```python
2 * A
```
so
```python
A * 2 == 2 * A
```
returns `True`. You can also do
```python
A / 2
```
but
```python
2 / A
```
is not meaningful.

Powers are defined for a square matrix to a scalar power (and for a scalar to a matrix power, but this latter requires eigenvalues so it hasn't been implemented).
```python
E ** 5
```
should give a sensible answer.

### REF and RREF
REF, or row echelon form, and RREF, or reduced row echelon form, are important parts of linear algebra. `B.ref()` and `B.rref()` will return the row echelon form of `B`, respectively. (From now on, I'll show commands inline for the sake of readability.)

### Transpose
`A.t()` returns the transpose of `A`.

### Nullspace and Column Space
`B.null()` returns the nullspace of `B`, while `B.col()` returns the column space. These vector spaces are displayed as a list of their basis vectors. The row space and left nullspace can be calculated using `B.t().col()` and `B.t().null()`, respectively---just as they are notated.
#### Basis and Dimension
For convenience, assign `S = B.t().col()`. `S.b()` returns the basis of `S`. `S.dim()` returns the dimesnion of `S`.

### Identity
`Matrix.i(4)` returns the 4x4 identity matrix, and so on.

### Inverse
Use `E.inv()` to invert `E`, a square matrix. `E * E.inv() == Matrix.i(3)` returns `True`.

`D` is singular, so `D.inv` gives a `NotInvertibleError`.

`A` is not square, so `A.inv()` raises `DimensionError`.

#### Left and right inverses
You can get the left and right inverses, if they exist, of a matrix `B` using `B.linv()` and `B.rinv()`. You can check that `B * B.rinv() == Matrix.i(3)`.

### Symmetry and Positive Definiteness
Use `E.symm()` to test whether `E` is symmetric. For any matrix, such as `B`, `B * B.t()` is symmetric. (You can test this by checking that `(B * B.t()).symm()` returns `True`.)

Use `(A * A.t()).posdef()` to test whether `A * A.t()` is positive definite.

### Orthogonality
To test whether two column vectors `A` and `B` are orthogonal, check that `A.ortho(B)` (or equivalently `B.ortho(A)`) returns `True`.

### Solving Systems
`A.solve(b)` will solve the equation *Ax = b* for a square matrix *A* and a column vector *b*.
#### Approximate Solve
For non-square matrices *A*, use `A.approxSolve(b)`, which minimizes the magnitude of the error vector *Ax - b* using a projection matrix.

### Determinant
`A.det()` gets the determinant of any square matrix `A`. Note that `D.det() == 0` because `D` is singular.

### Trace
`A.trace()` gets the trace of a square matrix. If you know the set of eigenvalues of a matrix (for my `E` it's *{-3.96506, 2.68306, 0.281995}*) you can verify that the sum is the trace and the product is the determinant.

## That's All, Folks
