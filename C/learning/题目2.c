#include <stdlib.h> // ststem
#include <string.h> // strcmp


int main()
{
    char input[20]= {0};  //存储数据
    //关机
    //system() - 专门用来之执行系统命令的
    system("shutdown -s -t 60"); //关机
again:
    printf("你是猪吗");
    scanf("%s",input);
    if (strcmp(input,"是")==0) //判断input中放的是不是"我是猪" - strcmp - string compare
    {
        system("shutdown -a");
    }
    else
    {
        goto again;
    }
    return 0;
}
