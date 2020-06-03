%{
#include <stdio.h>
#include <stdlib.h>
extern int yylex();
extern int yyparse();
extern FILE *yyin;
void yyerror(const char *s);
 
int count=0;

void printPartition(int n)
{
	int i=0;
	for(i=0;i<n;i++)
	{
		printf("-");
	}
	printf("\n");
}

%}

%union
{
	int val;
	char *sval;
	struct {
		int x;
		int y;
		int z;
	} input_layer_type;

	struct {
		int x;
		int y;
		int z;
		int num_kernels;
		int size_kernels;
		int weights;
	} conv_layer_type;

	struct {
		int input_size;
		int num_outputs;
		int weights;
	} dense_layer_type;
};

%type <input_layer_type> INPUT_LAYER;
%type <conv_layer_type> CONV_LAYER;
%type <conv_layer_type> conv_layers;
%type <dense_layer_type> DENSE_LAYER;

%token <sval> INPUT;
%token <sval> CONV;
%token <sval> DENSE;
%token <sval> ACTIVATIONFN;
%token <val> NUM;

%start model
%%
model: INPUT_LAYER conv_layers DENSE_LAYER 
{
	printf("\n");
	printPartition(50);
	printf("total Learnable Parameters : %d \n", $3.weights);
	printPartition(50);
	exit(0);
}
					

INPUT_LAYER: INPUT ':' '(' NUM ',' NUM ',' NUM ')' 
{ 
	$$.x = $4;
	$$.y = $6;
	$$.z = $8;

	printf("Network summary\n");
	printPartition(50);
	printf("Layer:%d Type:Input \t InputDims:(%d,%d,%d)\n",count,$$.x,$$.y,$$.z);
}
conv_layers: conv_layers CONV_LAYER
{
	$$.weights = $1.weights + $2.num_kernels*$2.size_kernels*$2.size_kernels+$2.num_kernels;
	$$.x = $1.x - $2.size_kernels + 1;
	$$.y = $1.y - $2.size_kernels + 1;
	$$.z = $2.num_kernels;


	if($$.x < 0 || $$.y < 0 || $$.y < 0)
	{
		printf("\nConv Layer expects 'positive' value but got 'negative' value\n");
		exit(0);
	}
}
						
conv_layers: CONV_LAYER 
{
	 $$.weights = $1.num_kernels*$1.size_kernels*$1.size_kernels + $1.num_kernels;
	 $$.x = $<input_layer_type>0.x - $1.size_kernels + 1;
	 $$.y = $<input_layer_type>0.y - $1.size_kernels + 1;
	 $$.z = $1.num_kernels;
}

CONV_LAYER: CONV ':' '(' NUM ',' NUM ',' ACTIVATIONFN ')' 
{ 

	$$.num_kernels = $4;
  	$$.size_kernels = $6;
  	count++;

	printf("Layer:%d Type:conv \t InputDims:(%d,%d,%d) \t LearnableParameter: %d \t Activation:%s\n",count,$<input_layer_type>0.x,$<input_layer_type>0.y,$<input_layer_type>0.z,$4*$6*$6+$4,$8);

	if($<input_layer_type>0.x < 0 || $<input_layer_type>0.y < 0 || $<input_layer_type>0.z < 0)
	{
		printf("Conv Layer expects 'positive' value but got 'negative' value\n");
		exit(0);
	}
}

DENSE_LAYER: DENSE ':' '(' NUM ',' ACTIVATIONFN ')' 
{ 
	$$.input_size = $<conv_layer_type>0.x*$<conv_layer_type>0.y*$<conv_layer_type>0.z;
	$$.weights = $<conv_layer_type>0.weights + 	$$.input_size*$4+$4;

	count++;

	printf("Layer:%d Type:Dense \t InputDims:(%d,%d,%d) \t LearnableParameter: %d \t Activation:%s\n",count,$<conv_layer_type>0.x,$<conv_layer_type>0.y,$<conv_layer_type>0.z,$$.input_size*$4+$4);

	if($<conv_layer_type>0.x <= 0 || $<conv_layer_type>0.y <= 0 || $<conv_layer_type>0.z <= 0)
	{
		printf("\nDense Layer expects 'non-zero' value \n");
		exit(0);
	}
}
%%

int main(int argv,char *argc[]) {

	yyin = fopen(argc[1], "r");

	FILE *fptr;
	fptr = fopen(argc[1], "r");

	//printPartition(50);
	printf("Neural Network Summary Generator\n\n");
	//printPartition(50);

	printPartition(50);
	printf("Input Network\n");
	printPartition(50);

	char c = fgetc(fptr); 
    while (c != EOF) 
    { 
        printf ("%c", c); 
        c = fgetc(fptr); 
    } 
	printf("\n\n");
    printPartition(50);
    //printf("\n");

    fclose(fptr);

	yyparse();
	return 0;
}

void yyerror(const char *s) {
  printf("parse error!  Message: %s\n",s);
  exit(-1);
}