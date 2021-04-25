#define NULL 0
#include <stdio.h>
/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* twoSum(int* nums, int numsSize, int target, int* returnSize){
    int i ;
    int j ;
    for(i = 0; i<numsSize;i++)
    { 
        for (int j = 1; j<= numsSize - i;j++)
        {
            if (nums[i]+ nums[i+j] == target)
            {
            // printf("%d%d\n",i,i+j);
             return i,i+j;
            }
        }
    }
}
int main()
{
    int nums[] = {2,7,11,15};
    int target = 9;
    int numsSize = sizeof(nums)/sizeof(nums[0]);
    int* returnSize = NULL;
   twoSum(nums,numsSize,target,returnSize);
}


