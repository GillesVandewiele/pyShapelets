#include "timeSeries.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

timeSeries::timeSeries(int n , double * y , int c , int samplingRate)
{
    this->length = n;
    this->c = c;
	this->originalClass = c;
    x = (double *)malloc(sizeof(double)*n);
	for( int i = 0 ; i < n ; i++ )
		x[i] = y[i];

	   d = 0;
	    strt = (int *)malloc(sizeof(int)*100);
	    lens = (int *)malloc(sizeof(int)*100);

    //memcpy(x,y,sizeof(double)*n);
    sumX = (double *)malloc(sizeof(double)*n);
    sumX2 = (double *)malloc(sizeof(double)*n);
    downSample(1);
   // computeX();
};


timeSeries::~timeSeries()
{
	
    free(x); 
	free(strt);
	free(lens);
	
};

int timeSeries::downSample(int s)
{
	this->samplingRate = s;
    original_x = (double *)malloc(sizeof(double)*this->length);
	for( int i = 0 ; i < this->length ; i++ )
		original_x[i] = x[i];

	
	for( int i = 0 ; i < this->length ; i+=s )
		x[i] = original_x[i];
	this->originalLength = this->length;
	this->length /= s;
	computeX();
	return s;
}

void timeSeries::computeX()
{
    sumX[0] = x[0];
    sumX2[0] = x[0]*x[0];

    for(int k = 1 ; k < length ; k++ )
    {
        sumX[k] = sumX[k-1]+x[k];
        sumX2[k] = sumX2[k-1]+x[k]*x[k];
    }     
};

double timeSeries::mean(int i , int len)
{
    return (sumX[i+len-1]-sumX[i]+x[i])/len;
};


double timeSeries::stdv(int i , int len)
{
	double mu = (sumX[i+len-1]-sumX[i]+x[i])/len;
	double s2 = ((sumX2[i+len-1]-sumX2[i]+x[i]*x[i])/len)-mu*mu;
	if( s2 <= 0 )
		return 0;
	else
		return sqrt(s2);
};


int timeSeries::insertPos(int i , int len)
{
	strt[d] = i;
	lens[d++] = len;
    return d;
};

int timeSeries::checkPos(int s , int len)
{
	for(int i = 0; i < d ; i++ )
	{
		if( s < strt[i] && strt[i] <= s+len )
			return 0;

		else if( s >= strt[i] && s <= strt[i]+len )
		    return 0;

	}
	return 1;
};

void timeSeries::clearPos()
{
	d = 0;
};


void timeSeries::init()
{
	/*
         insertPos(76,3);
         insertPos(123,3);
         insertPos(168,3);
         insertPos(214,3);
         insertPos(261,3);
		 insertPos(307,3);*/
};
