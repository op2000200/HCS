#include <gtest/gtest.h>
#include <iostream>
#include <kernel.cu>

TEST(SumTest, OnePlusOneEqualsTwoCPU) {
  EXPECT_EQ(sumH(1,1), 2);
}

__managed__ float a[32]; 
__managed__ float b[32];
__managed__ float c[32];

TEST(SumTest, OnePlusOneEqualsTwoCUDA) {

	
	a[0] = 1.f;
	b[0] = 1.f;
	
	sum<<<1, 32>>>(a, b, c);
	cudaDeviceSynchronize();
	
	EXPECT_EQ(c[0], 2.f);
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}