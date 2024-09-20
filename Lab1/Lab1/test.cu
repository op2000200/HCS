#include <gtest/gtest.h>
#include <iostream>
#include <kernel.cu>

TEST(CPUTest, Test1) {
  EXPECT_EQ(sumH(1,1), 2);
}

TEST(CPUTest, Test2) {
  EXPECT_EQ(sumH(2.045, 1.005), 2.05);
}

TEST(CPUTest, Test3) {
  EXPECT_EQ(sumH(1000.1000, 2000.2000), 3000.3000);
}

__managed__ float a[1]; 
__managed__ float b[1];
__managed__ float c[1];

TEST(CUDATest, OnePlusOneEqualsTwo) {
	a[0] = 1.f;
	b[0] = 1.f;
	
	sum<<<1, 1>>>(a, b, c);
	cudaDeviceSynchronize();
	
	EXPECT_EQ(c[0], 2.f);
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}