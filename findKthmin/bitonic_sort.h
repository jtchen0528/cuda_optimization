
/* ----------------------- Host prototypes -----------------------------*/
void Kernel_driver(int a[], int n, int blk_ct, int th_per_blk);
void Usage(char* prog_name);
void Get_args(int argc, char *argv[], int *n_p, int *blk_ct, int *th_per_blk, int* mod_p);
void Generate_list(int a[], int n, int mod);
void Print_list(int a[], int n, char* title);
void Read_list(int a[], int n);
int  Check_sort(int a[], int n);
void Print_unsigned(unsigned val, unsigned field_width);
unsigned Get_width(unsigned val);