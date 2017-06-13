#include<iostream>
#include<cmath>
using namespace std;/*  sieve of ERASTHOSTHENES               */
int main(){
int N=10001;
bool prime[N];
int n=sqrt(N);
//cout<<n;
for(int i=2;i<N;i+=2){
    prime[i]=1;
    }
for(int i=3;i<n;i+=2)
{if(prime[i]==0){
for(int j=2*i;j<N;j+=i){
prime[j]=1;
}
}
}


prime[1]=1;
for(int i=1;i<N;i++){
    if(prime[i]==0){
        cout<<i<<" ";

    }


}

}
