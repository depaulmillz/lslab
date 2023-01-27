//
// Created by depaulsmiller on 9/3/20.
//

#include <vector>
#include <lslab/map.h>
#include <thrust/host_vector.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <unistd.h>

void usage(char* exec) {
    std::cout << "Usage: " << exec << " [-s <size_log_2>] [-b <batch_size>] [-p <population_size>] [-r <range>] [-c <threads_per_cta>]" << std::endl;
}

int main(int argc, char** argv) {

    int size_log_2 = 20;
    int batch_size = 1 << 10;
    int popsize = (1 << size_log_2) / 2;
    int range = 2 * (1 << size_log_2);
    int cta_size = 256;

    char c;
    while((c = getopt(argc, argv, "s:b:p:r:c:h")) != -1) {
        switch(c) {
            case 'c':
                cta_size = atoi(optarg);
                break;
            case 'p':
                popsize = atoi(optarg);
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            case 'r':
                range = atoi(optarg);
                break;
            case 's':
                size_log_2 = atoi(optarg);
                if(size_log_2 >= 32) {
                    return 1;
                }
                break;
            default:
                usage(argv[0]);
                return 0;
        }
    }


    int size = 1 << size_log_2;

    lslab::map<unsigned long long, unsigned> m(size_log_2);

    std::cerr << "Populating" << std::endl;

    thrust::host_vector<unsigned long long> keys(popsize);

    thrust::sequence(keys.begin(), keys.end(), 1);

    thrust::device_vector<unsigned long long> keys_device = keys;
    thrust::device_vector<unsigned> values_device(keys_device.size());
    thrust::device_vector<unsigned> results_device(keys_device.size());

    unsigned long long* k_d = keys_device.data().get();
    unsigned* v_d = values_device.data().get();
    unsigned* r_d = results_device.data().get();

    m.put(k_d, v_d, r_d, keys.size());

    std::cerr << "Populated" << std::endl;

    std::cout << "---- Get -----" << std::endl;

    for (int rep = 0; rep < 10; rep++) {

        thrust::host_vector<unsigned long long> keys(batch_size);
        thrust::device_vector<unsigned long long> keys_device(batch_size);
        thrust::device_vector<cuda::std::pair<bool, unsigned>> values_device(batch_size);
        
        for (int i = 0; i < batch_size; ++i) {
            unsigned long long key = static_cast<unsigned long long>(rand() / (double) RAND_MAX * (range));
            keys[i] = key;
        }

        keys_device = keys;

        gpuErrchk(cudaProfilerStart());
        auto start = std::chrono::high_resolution_clock::now();
        switch(cta_size) {
            case 32:
                m.get<32>(keys_device.data().get(), values_device.data().get(), keys_device.size());
                break;
            case 64:
                m.get<64>(keys_device.data().get(), values_device.data().get(), keys_device.size());
                break;
            case 128:
                m.get<128>(keys_device.data().get(), values_device.data().get(), keys_device.size());
                break;
            case 256:
                m.get<256>(keys_device.data().get(), values_device.data().get(), keys_device.size());
                break;
            case 512:
                m.get<512>(keys_device.data().get(), values_device.data().get(), keys_device.size());
                break;
            case 1024:
                m.get<1024>(keys_device.data().get(), values_device.data().get(), keys_device.size());
                break;
            default:
                std::cerr << "Size not supported" << std::endl;
                return 1;
        }
        gpuErrchk(cudaStreamSynchronize(0x0));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        gpuErrchk(cudaProfilerStop());

        std::cout << "Standard Uniform test" << std::endl;
        std::cout << "Latency\t" << dur.count() * 1e3 << " ms" << std::endl;
        std::cout << "Throughput\t" << batch_size / dur.count() / 1e6 << " Mops" << std::endl;
    }

    std::cout << "---- Put -----" << std::endl;

    for (int rep = 0; rep < 10; rep++) {

        thrust::host_vector<unsigned long long> keys(batch_size);
        thrust::device_vector<unsigned long long> keys_device(batch_size);
        thrust::device_vector<unsigned> values_device(batch_size);
        thrust::device_vector<unsigned> results_device(batch_size);
        
        for (int i = 0; i < batch_size; ++i) {
            unsigned long long key = static_cast<unsigned long long>(rand() / (double) RAND_MAX * (range));
            keys[i] = key;
        }

        keys_device = keys;

        gpuErrchk(cudaProfilerStart());
        auto start = std::chrono::high_resolution_clock::now();
        switch(cta_size) {
            case 32:
                m.put<32>(keys_device.data().get(), values_device.data().get(), results_device.data().get(), keys_device.size());
                break;
            case 64:
                m.put<64>(keys_device.data().get(), values_device.data().get(), results_device.data().get(), keys_device.size());
                break;
            case 128:
                m.put<128>(keys_device.data().get(), values_device.data().get(), results_device.data().get(), keys_device.size());
                break;
            case 256:
                m.put<256>(keys_device.data().get(), values_device.data().get(), results_device.data().get(), keys_device.size());
                break;
            case 512:
                m.put<512>(keys_device.data().get(), values_device.data().get(), results_device.data().get(), keys_device.size());
                break;
            case 1024:
                m.put<1024>(keys_device.data().get(), values_device.data().get(), results_device.data().get(), keys_device.size());
                break;
            default:
                std::cerr << "Size not supported" << std::endl;
                return 1;
        }
        gpuErrchk(cudaStreamSynchronize(0x0));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        gpuErrchk(cudaProfilerStop());

        std::cout << "Standard Uniform test" << std::endl;
        std::cout << "Latency\t" << dur.count() * 1e3 << " ms" << std::endl;
        std::cout << "Throughput\t" << batch_size / dur.count() / 1e6 << " Mops" << std::endl;
    }
    return 0;
}
