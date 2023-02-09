#include <iostream>
#include <chrono>
#include <cublas_v2.h>

constexpr unsigned max_log_N = 15;
constexpr unsigned min_log_N = 7;

constexpr unsigned test_count = 1u << 7;

int main() {
	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	double *A_ptr, *x_ptr, *y_ptr;
	const auto max_N = 1lu << max_log_N;
	cudaMalloc(&A_ptr, sizeof(double) * max_N * max_N);
	cudaMemset(A_ptr, 0, sizeof(double) * max_N * max_N);
	cudaMalloc(&x_ptr, sizeof(double) * max_N);
	cudaMemset(x_ptr, 0, sizeof(double) * max_N);
	cudaMalloc(&y_ptr, sizeof(double) * max_N);
	cudaMemset(y_ptr, 0, sizeof(double) * max_N);

	std::printf("N,bandwidth_in_tbyteps,throughput_in_tflops\n");

	const double alpha = 1., beta = 0.;
	for (unsigned log_N = min_log_N; log_N <= max_log_N; log_N++) {
		const auto N = 1lu << log_N;
		cublasDgemv(cublas_handle, CUBLAS_OP_N, N, N, &alpha, A_ptr, N, x_ptr, 1, &beta, y_ptr, 1);

		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();

		for (unsigned i = 0; i < test_count; i++ ) {
			cublasDgemv(cublas_handle, CUBLAS_OP_N, N, N, &alpha, A_ptr, N, x_ptr, 1, &beta, y_ptr, 1);
		}

		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();
		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;

		std::printf("%lu,%e,%e\n",
				N,
				(N * N + 2 * N) * sizeof(double) / elapsed_time * 1e-12,
				(2 * N * N) / elapsed_time * 1e-12
				);
	}

	cudaFree(A_ptr);
	cudaFree(x_ptr);
	cudaFree(y_ptr);
}
