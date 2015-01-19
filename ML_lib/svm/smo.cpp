
/*
   label: array
   data:  sample by feature
   # names in short
   bnd: bound example, i.e. alpha != 0 and C
   nbnd: non-bound example, i.e. alpha  = 0 or  C
   -- continous --
*/

#ifndef Rand
 #define SEED 300 
#endif
#ifndef TOL
 #define TOL 0.0001
#endif

int Rand(seed)
{
 srand(seed);
 return rand();
}


class svm
{
 public:
       svm(double C=0.0): C(C_) label(NULL), data(NULL), alpha(NULL), w(NULL), b(NULL), bnd(NULL), nbnd(NULL), E(NULL) {} 
       svm& fit(std::vector<double>& ,std::vector<int>& ) 
       int predict(std::vector<double>& )
 private:
       double C_;
       std::vector<int> label_;
       std::vector<double> data_;
       std::vector<double> alpha_;
       std::vector<double> w_;
       double b_;
       std::list<unsigned int> bnd_, nbnd_;
       std::vector<double> E_;

       /*--------- init --------*/
       init_alpha(std::vector<double>& );
       void init_bnd_nbnd(std::vector<double>& ,std::vector<double>& );
       void init_E(std::vector<double>& );
       
       /*--------- SMO --------*/
       bool smo_KKT(double, int, double);
       int  smo_KKT_loop(std::vector<unsigned int>& );
       unsigned int smo_get_a1();
       unsigned int smo_get_a2();      
       void smo_update_a12(unsigned int ,unsigned int );
       void smo_main();
}

svm& svm::fit(std::vector<double>& data,std::vector<int>& label)
{
 int nsample;
 this.data_ = data;
 this.label_ = label;
 nsample = label.size();
 init_alpha(this.alpha_, nsample);
 init_bnd_nbnd(this.bnd_, this.nbnd_);
 init_E(this.E_); 
 smo_main(); 
}

int svm::predict(std::vector<double>& X)
{
 double y = 0;
 for(int i=0;i<this.alpha_.size();++i)
    y += this.alpha_[i] * this.label[i] * Kernel(X,this.data_[i]);
 y += this.b_;
 return y; 
}

double Kernel(std::vector<double>& X1, std::vector<double>& X2)
{
 double sum=0.0;
 for(int i=0;i!=X1.size();++i)
    sum += X1[i] * X2[i];
}

void svm::init_alpha(std::vector<double>& alpha, int nsample=0)
{
 if(nsample==0)
   return;
 for(int i=0;i<nsample;++i)
    {
     alpha.push_back(0.0); 
    }
}
void svm::init_bnd_nbnd(std::vector<unsigned int>& bnd, std::vector<unsigned int>& nbnd)
{
 for(int i=0;i<this.alpha_.size();++i)
    {
     if(this.alpha_[i]==0||this.alpha_[i]==this.C)
       bnd.push_back(i);
     else
       nbnd.push_back(i);
    }
}
void svm::init_E(std::vector<double>& E)
{
 /*
   Required: alpha, data, label
 */
 double e = 0.0;
 int nsample = this.alpha_.size();
 for(int i=0;i<nsample;++i)
    {
     for(int j=0;j<nsample;++j)
        e += this.alpha_[j] * this.label_[j] *  Kernel(this.data_[i],this.data_[j]); // fix me  
    }
    e = e + this.b_ - label_[i];
    E.push_back(e);
}

bool svm::smo_KKT(double a, int y, double corrected_error)
{
 // '0' and '1' for 'not satify KKT' or 'satify KKT'
 if(a<C && y*corrected_error<-1*TOL)||(a>0 && y*corrected_error>TOL)
   return 0;
 return 1; 
}

unsigned int svm::smo_get_a1()
{
 /*
   Obj: 
   2-Step Algo:
     step1, look within bound samples;
     step2, look within non-bound samples;
   ReturnVal:
     if no positive progress, then return index -1; else, return index i2
 */
        
 int bnd_n = this.bnd_.size();
 int nbnd_n = this.nbnd_.size();
 
 for(int iter=0,currp=Rand()%(bnd_n-0)+0; iter!=bnd_n;++iter,++currp) // start at a random point 
    {
        currp = currp%(bnd_n-0)+0;
        idx = self.bnd[currp];
         
        a             = this.alpha_[idx];
        y             = this.label_[idx];
        x             = this.data_[idx,:] # fix me
        corrected_e = y * this.E_[idx]; 
        if(!smo_KKT(a,y,corrected_e)) // not satisfy
          return idx;
    } 
    
 for(int iter=0,currp=Rand()%(nbnd_n-0)+0; iter!=nbnd_n;++iter,++currp)
    {
        currp = currp%(nbnd_n-0)+0;
        idx = self.nbnd[currp];

        a   = this.alpha_[idx];
        y   = this.label_[idx];
        x   = this.data_[idx,:] # fix me
        corrected_e = y * this.E_[idx]; 
        if(!smo_KKT(a,y,corrected_e)) // not satisfy
          return idx;
    }
 return -1;
}

unsigned int svm::get_a2(int i1)
{
 int bnd_n = this.bnd_.size();
 int nbnd_n = this.nbnd_.size();
 unsigned int i2;
 double selE; // selected E
 /*
   Obj: find selE which maximize 'E1 - E2' 
   2-Step Algo:
     step1, look within bound samples;
     step2, look within non-bound samples;
   ReturnVal:
     if no positive progress, then return index -1; else, return index i2
 */
 for(int iter=0,currp=Rand()%(bnd_n-0)+0; iter!=bnd_n;++iter,++currp) // start at a random point 
    {
        currp = currp%(bnd_n-0)+0;
        idx = self.bnd_[currp];
        if(iter==0||(this.E_[i1]>0 && this.E_[idx]<selE)||(this.E_[i1]<0 && this.E_[idx]>selE)) // if E[i1]>0 ,then find min E; else, find max E
          {
           i2 = idx;
           selE = this.E_[i2];
          }
        if(E1-selE>0)
          return i2;
    } 
    
 for(int iter=0,currp=Rand()%(nbnd_n-0)+0; iter!=nbnd_n;++iter,++currp)
    {
        currp = currp%(nbnd_n-0)+0;
        idx = self.nbnd[currp];
        if((this.E_[i1]>0 && this.E_[idx]<selE)||(this.E_[i1]<0 && this.E_[idx]>selE))
          {
           i2 = idx;
           selE = this.E_[i2];
          }
    }
   return (E1-selE>0)?i2:-1; // if no positive progress, then return -1
}


void svm::smo_update_a12(unsigned int i1,unsigned int i2)
{
 // untrimmed a2
 double K11 = Kernel(self.data[i1,:],self.data[i1,:]);
 double K22 = Kernel(self.data[i2,:],self.data[i2,:]);
 double K12 = Kernel(self.data[i1,:],self.data[i2,:]);
 double eta = (K11 + K22 - 2*K12);
 double a1 = this.alpha_[i1]; double a2 = this.alpha_[i2];
 int    y1 = this.label_[i1]; int    y2 = this.label_[i2];
 double a1_new,                a2_new;
 double b1_new_i1,             b2_new_i2;
 double L,H;
 double C = this.C;
 double f1,f2,L1,H1,Psy1,Psy2;

 /* L & H */ 
 if(y1!=y2)
   {L = max(0,a2-a1);   H = min(C,C+a2-a1);} 
 else
   {L = max(0,a2+a1-C); H = min(C,a2+a1);}
 // if L == H, then return
 if(L==H)
   return;
 // if eta <=0, then return
 if(eta<=0)
   {
    f1 = y1 * (E) 
    return;
   }
 
 // a2_new && a1_new 
 a2_new = a2 + (this.E_[i1] - this.E_[i2])/eta; // update a2
 a2_new = max(L,a2_new); a2_new = min(H,a2_new); // trimmed a2 
 if(abs(a2_new-a2)<0.00001) // if no significant change, then return
   return; 
 a1_new = (a2 - a2_new) * y1 * y2 + a1; // update a1
 
 /* update alpha, b, E, bnd, nbnd */
 // alpha
 this.alpha_[i2] = a2_new; this.alpha_[i1] = a1_new; 
 // b 
 b_new_i1 = -1 * this.E_[i1] - y1 * K11 * (a1_new - a1) - y2 * K21 * (a2_new - a2) + b;
 b_new_i2 = -1 * this.E_[i2] - y1 * K12 * (a1_new - a1) - y2 * K22 * (a2_new - a2) + b;
 this.b_ = ((a1_new==0||a1_new==C)&&(a2_new==0||a2_new==C))?(b_new_i1/2.0+b_new_i2/2.0):b_new_i1;
 // E
 for(int i=0;i!=E.size();i)
    this.E_[i] = predict(this.data_[i]) - this.label_[i]; 
 // bnd & nbnd
 if((a1==0||a1==C)&&!(a1_new==0||a1_new==C))
   {
    this.bnd_.remove(i1); 
    this.nbnd_.push_back(i1);
   }
 else if(!(a1==0||a1==C)&&(a1_new==0||a1_new==C))
   {
    this.nbnd_.remove(i1); 
    this.bnd_.push_back(i1);
   }
 if((a2==0||a2==C)&&!(a2_new==0||a2_new==C))
   {
    this.bnd_.remove(i2); 
    this.nbnd_.push_back(i2);
   }
 else if(!(a2==0||a2==C)&&(a2_new==0||a2_new==C))
   {
    this.nbnd_.remove(i2); 
    this.bnd_.push_back(i2);
   }

 return;
}

void svm::smo_main()
{
 //
 int stop_flag;
 int iter = 0;
 unsigned int i1, i2;
 double a1, a2;
 while(1)
      {
       stop_flag = 1;
       while(iter++<this.label.size()) // loop through examples
            {  
             // first alpha
             i1 = get_a1(bnd,nbnd);
             if(i1==-1)
               continue;
             stop_flag = 0;
             a1 = this.alpha_[i1];
             // second alpha
             i2 = get_a2(bnd,nbnd); 
             if(i2==-1) // no positive progress
               continue;
             a2 = this.alpha_[i2];
             // update a1 & a2
             update_a12(i1,i2);
            }
       if(stop_flag)
         break;
      }
 return;
}
