using System.Runtime.InteropServices;

namespace ANN
{
    /// <summary>
    /// Wrapper for Basic Linear Algebra Subprograms (BLAS) operations using the Intel Math Kernel Library (MKL).
    /// Provides methods for matrix and vector operations commonly used in neural network computations.
    /// </summary>
    public class Blas
    {
        /// <summary>
        /// Performs a single-precision matrix-vector multiplication.
        /// Y = alpha * A * X + beta * Y
        /// </summary>
        /// <param name="Order">Specifies the data ordering. Typically row-major or column-major.</param>
        /// <param name="TransA">Indicates whether to transpose matrix A.</param>
        /// <param name="M">Number of rows in matrix A.</param>
        /// <param name="N">Number of columns in matrix A.</param>
        /// <param name="alpha">Scalar multiplier for matrix A.</param>
        /// <param name="A">The matrix A in flattened array form.</param>
        /// <param name="lda">Leading dimension of A (number of elements in each row or column, depending on Order).</param>
        /// <param name="X">Input vector X.</param>
        /// <param name="incX">Stride within vector X. Typically set to 1.</param>
        /// <param name="beta">Scalar multiplier for vector Y.</param>
        /// <param name="Y">Output vector Y, which also serves as the initial value.</param>
        /// <param name="incY">Stride within vector Y. Typically set to 1.</param>
        [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cblas_sgemv(int Order, int TransA, int M, int N, float alpha, float[] A, int lda, float[] X, int incX, float beta, float[] Y, int incY);

        /// <summary>
        /// Performs a single-precision matrix-matrix multiplication.
        /// C = alpha * A * B + beta * C
        /// </summary>
        /// <param name="Order">Specifies the data ordering. Typically row-major or column-major.</param>
        /// <param name="TransA">Indicates whether to transpose matrix A.</param>
        /// <param name="TransB">Indicates whether to transpose matrix B.</param>
        /// <param name="M">Number of rows in matrix A and C.</param>
        /// <param name="N">Number of columns in matrix B and C.</param>
        /// <param name="K">Number of columns in matrix A and rows in matrix B.</param>
        /// <param name="alpha">Scalar multiplier for the product of matrices A and B.</param>
        /// <param name="A">Matrix A in flattened array form.</param>
        /// <param name="lda">Leading dimension of A (number of elements in each row or column, depending on Order).</param>
        /// <param name="B">Matrix B in flattened array form.</param>
        /// <param name="ldb">Leading dimension of B.</param>
        /// <param name="beta">Scalar multiplier for matrix C.</param>
        /// <param name="C">Matrix C, which stores the result of the operation.</param>
        /// <param name="ldc">Leading dimension of C.</param>
        [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc);

        /// <summary>
        /// Element-wise vector multiplication.
        /// y[i] = a[i] * x[i]
        /// </summary>
        /// <param name="n">Number of elements to multiply.</param>
        /// <param name="a">Input vector a.</param>
        /// <param name="x">Input vector x.</param>
        /// <param name="y">Output vector y, which stores the result of the element-wise multiplication.</param>
        [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void vsMul(int n, float[] a, float[] x, float[] y);

        /// <summary>
        /// Performs the AXPY operation: adds a scaled vector to another vector.
        /// y = alpha * x + y
        /// </summary>
        /// <param name="n">Number of elements in vectors x and y.</param>
        /// <param name="alpha">Scalar multiplier for vector x.</param>
        /// <param name="x">Input vector x.</param>
        /// <param name="incx">Stride within vector x. Typically set to 1.</param>
        /// <param name="y">Input/output vector y, which stores the result of the operation.</param>
        /// <param name="incy">Stride within vector y. Typically set to 1.</param>
        [DllImport("mkl_rt.2.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void cblas_saxpy(int n, float alpha, float[] x, int incx, float[] y, int incy);
    }
}
