#include <gtest/gtest.h>
#include <iostream>
#include <kernel.cu>

TEST(CPUTest, Test1) {
  EXPECT_FLOAT_EQ(sumH(1,1), 2);
}

TEST(CPUTest, Test2) {
  EXPECT_FLOAT_EQ(sumH(2.045, 1.005), 3.05);
}

TEST(CPUTest, Test3) {
  EXPECT_FLOAT_EQ(sumH(1000.1000, 2000.2000), 3000.3000);
}

__managed__ float a[1]; 
__managed__ float b[1];
__managed__ float c[1];

TEST(CUDATest, OnePlusOneEqualsTwo) {
	a[0] = 1.f;
	b[0] = 1.f;
	
	sum<<<1, 1>>>(a, b, c);
	cudaDeviceSynchronize();
	
	EXPECT_FLOAT_EQ(c[0], 2.f);
}

__managed__ float a2[4] {1.0f, 2.0f, 3.0f, 4.0f}; 
__managed__ float b2[4] {5.0f, 6.0f, 7.0f, 8.0f};
__managed__ float c2[4];

TEST(CUDATest, ArrTest) {
	sum<<<1, 4>>>(a2, b2, c2);
	cudaDeviceSynchronize();
	
	EXPECT_FLOAT_EQ(c2[0], 6.f);
	EXPECT_FLOAT_EQ(c2[1], 8.f);
	EXPECT_FLOAT_EQ(c2[2], 10.f);
	EXPECT_FLOAT_EQ(c2[3], 12.f);
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}