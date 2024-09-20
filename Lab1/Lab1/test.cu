#include <gtest/gtest.h>
#include <iostream>
#include <kernel.cu>

TEST(SumTest, OnePlusOneEqualsTwoCPU) {
  EXPECT_EQ(sumH(1,1), 2);
}

TEST(SumTest, OnePlusOneEqualsTwoCUDA) {
	const float* a;
	const float* b;
	
	a = 1.f;
	b = 1.f;
	
	float* c;
	sum(a,b,c);
	EXPECT_EQ(c, 2.f);
}


int main(int argc, char *argv[]) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}