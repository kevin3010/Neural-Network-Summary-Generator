%{
	#include "./architecture-summary.tab.h"
	#include <stdio.h>
%}
%option noyywrap
 

%%
[ \t\n] {}
[-]?[0-9]+            { yylval.val = atoi(yytext); return NUM; }
"input" 		{ yylval.sval = strdup(yytext); return INPUT; }
"conv" 		{ yylval.sval = strdup(yytext); return CONV; }
"dense" 		{ yylval.sval = strdup(yytext); return DENSE; }
"sigmoid"|"relu"|"softmax"|"tanh"      { yylval.sval = strdup(yytext); return ACTIVATIONFN;}
. return yytext[0];
%%


