import numpy as np
import random
import copy

import scipy.linalg
import scipy.optimize

class simplex:   
    
    def simplexFunction(self,A,c,b,HowToSolve,Phase,B = None,N = None):
        self.constraintMatrix = A
        self.objCoefficient = c
        self.rightVector = b
        Basis_num = self.constraintMatrix.shape[0]
        dim_x = self.constraintMatrix.shape[1]
        Nonbasis_num = dim_x - Basis_num 
        x_star = np.zeros(dim_x).reshape(dim_x,1) #解を入れる空配列
        iter_num = 0 #反復回数測定用
        M = self.constraintMatrix #制約行列
        print(M[:,:-Basis_num])
        print(np.linalg.matrix_rank(M[:,:-Basis_num]) == len(M[:,:-Basis_num]))
        #初期基底・非基底添字を決定
        if Phase:
            #フェーズ２
            index_bases = B
            index_notbases = N
        else:
            
            #フェーズ１
            index_bases = np.arange(Nonbasis_num,dim_x)
            index_notbases = np.arange(Nonbasis_num)

        print(index_bases,index_notbases)
   
        while True:    
            # print(index_bases)
            c_base = self.objCoefficient[index_bases,:] #基底係数
            c_notbase = self.objCoefficient[index_notbases,:] #非基底係数
            Base = M[:,index_bases] #基底行列
            print(Base)
            print(np.linalg.matrix_rank(Base) == len(Base))
            Notbase = M[:,index_notbases] #非基底行列
          
            if not np.linalg.matrix_rank(Base) == len(Base):
                #基底行列が非正則
                return None
            else:
                multipler = np.linalg.solve(Base.T, c_base) #単体乗数
                costFunction = c_notbase - Notbase.T @ multipler #スラック変数、コスト関数
                b_childa = np.linalg.solve(Base,self.rightVector).reshape(Basis_num,1) #基底変数の定数項
            
                #最適性条件判定（コスト関数>=0)を満たすか
                if np.all(costFunction >= 0): 
                    
                    #最適性満足
                    x_star[index_bases] = b_childa 
                    x_star[index_notbases] = np.zeros(Nonbasis_num).reshape(Nonbasis_num,1)
                    opt_val = np.dot(x_star.T,self.objCoefficient).item()
                    return (x_star,opt_val,index_bases,index_notbases,iter_num)
                else:
                #負の係数が存在
                    if HowToSolve:
                        #最小添字規則で実行
                        index_in = np.where(costFunction < 0)[0][0] #コスト関数が負の値をとる最小添字 = 入る変数
                        col_index_in =np.linalg.solve(Base, Notbase[:,index_in:index_in + 1]).reshape(Basis_num,1) #非基底列ベクトルの最小添字に対応する列
                    else:
                        #最大係数規則で実行
                        index_in = np.argmax(-1 * costFunction) #コスト関数の最大係数を与える添字＝入る変数
                        col_index_in = np.linalg.solve(Base, Notbase[:,index_in:index_in + 1]).reshape(Basis_num,1) #非基底列ベクトルの最大係数添字に対応する列
                        
                    #　有界性判定（コスト関数が負の添字のピボット>0)を満たすか
                    if  np.all(col_index_in <= 0): 
                        #解は無限小
                        x_star = None
                        opt_val = None
                        break
                    else:
                        #(A~B^-1)*b / (A~B^-1)*A~N~kの最小値を見つける
                        ratio  = np.zeros(Basis_num)
                        for i in range(len(col_index_in)):
                            if col_index_in[i] > 0:
                                ratio[i] = b_childa[i] / col_index_in[i] 
                            else:
                                ratio[i] = np.inf
                        
                        index_out = np.argmin(ratio)
                        
                        #ピボット変換
                        index_notbases[index_in], index_bases[index_out] = index_bases[index_out], index_notbases[index_in]
                        iter_num += 1 #反復回数測定                                   

        
class simplex_phase1:
    def __init__(self,A,b):
        self.constraintMatrix = A
        self.rightVector = b
        
    def phase1Function(self):
        dimx_constMat = self.constraintMatrix.shape[0]
        dimy_constMat = self.constraintMatrix.shape[1]
        subprobMatrix = np.concatenate((self.constraintMatrix,np.identity(dimx_constMat)),axis=1) #第1段階の制約行列
        subprobCoefficient = np.concatenate((np.zeros(dimy_constMat).reshape(-1,1),np.ones(dimx_constMat).reshape(-1,1)),axis=0) #目的関数＝人工変数の和
        artificial_index = np.arange(dimy_constMat,dimx_constMat + dimy_constMat) #人工変数の添字
        
        #bに負の値が含まれる場合
        if not np.all(self.rightVector >= 0):
            neg_index = np.where(self.rightVector < 0)[0]
            self.rightVector[neg_index] = -1 * self.rightVector[neg_index]
            self.constraintMatrix[neg_index] = -1 * self.constraintMatrix[neg_index]
            
                    
        #フェーズ1の最適解を求める
        solution_phase1 = simplex()
        sol_phase1 = solution_phase1.simplexFunction(subprobMatrix,subprobCoefficient,self.rightVector,False,False)
        if sol_phase1 is None:
            return None
        else:
            
            value_phase1sol = sol_phase1[1]
            baseIndex_phase1sol = np.array(sol_phase1[2])
            nonbaseIndex_phase1sol = np.array(sol_phase1[3])
            
            print(subprobCoefficient)
            print(subprobMatrix)
            print(value_phase1sol)
            print(baseIndex_phase1sol)
            print(nonbaseIndex_phase1sol)
    
        
        #実行可能性判定（人工変数＝0を満たすか)
        if value_phase1sol > 0:
            return None
        else:
            #基底添字集合に人工変数が入っていないか
            if np.all(np.isin(artificial_index,nonbaseIndex_phase1sol)):
                NonbasisNotArtif = np.isin(nonbaseIndex_phase1sol,artificial_index,invert=True)
                nonbaseIndexForphase2 = nonbaseIndex_phase1sol[NonbasisNotArtif]
                baseIndexForphase2 = baseIndex_phase1sol
                print(baseIndexForphase2,nonbaseIndexForphase2)
                return (baseIndexForphase2,nonbaseIndexForphase2)
            else:
                #     # 人工変数が基底集合に残っている場合の処理
                # for art_idx in artificial_index:
                #     if art_idx in baseIndex_phase1sol:
                #         # 人工変数を基底から非基底に移す処理
                #         for non_art_idx in nonbaseIndex_phase1sol:
                #             if non_art_idx not in artificial_index:
                #                 # 基底変数と非基底変数を交換
                #                 idx_base = np.where(baseIndex_phase1sol == art_idx)[0][0]
                #                 idx_nonbase = np.where(nonbaseIndex_phase1sol == non_art_idx)[0][0]
                #                 baseIndexForphase2[idx_base], nonbaseIndexForphase2[idx_nonbase] = (
                #                     nonbaseIndexForphase2[idx_nonbase],
                #                     baseIndexForphase2[idx_base],
                #                 )
                #                 # 入れ替え後の基底行列が正則か確認
                #                 if np.linalg.matrix_rank(subprobMatrix[:, baseIndex_phase1sol]) == len(baseIndex_phase1sol):
                #                     break
                                
                                
                                    
                #基底添字集合に入っている人工変数を枢軸変換
                bidx1_copy = copy.deepcopy(baseIndex_phase1sol)
                nidx1_copy = copy.deepcopy(nonbaseIndex_phase1sol)
                baseIndexForphase2 = copy.deepcopy(baseIndex_phase1sol)
                nonbaseIndexForphase2 = copy.deepcopy(nonbaseIndex_phase1sol)
                r , ridx = np.intersect1d(artificial_index,bidx1_copy, return_indices=True)
                s , sidx = np.setdiff1d(nidx1_copy,artificial_index, return_indices = True)
                
                #人工変数をひとつずつ入れ替え
                for i in range(len(r)):
                    for j in range(len(s)):
                        #基底変数になっている人工変数と非基底変数非人工変数を入れ替え
                        baseIndexForphase2[ridx[i]],nonbaseIndexForphase2[sidx[j]] = nonbaseIndexForphase2[sidx[j]],baseIndexForphase2[ridx[i]]
                        #入れ替え後の基底行列の正則性判定
                        if np.linalg.matrix_rank(subprobMatrix[:,baseIndexForphase2]) == len(baseIndexForphase2):
                            #正則なので入れ替え完了
                            break
                        else:
                            #もとに戻して違う非基底変数非人工変数で試す
                            baseIndexForphase2[ridx[i]],nonbaseIndexForphase2[sidx[j]] = nonbaseIndexForphase2[sidx[j]],baseIndexForphase2[ridx[i]]
                
                #すべての人工変数が非基底変数になったので取り除く
                NonbasisNotArtif = np.isin(nonbaseIndexForphase2,artificial_index,invert=True)
                nonbaseIndexForphase2 = nonbaseIndexForphase2[NonbasisNotArtif]
                print(baseIndexForphase2,nonbaseIndexForphase2)
                return (baseIndexForphase2,nonbaseIndexForphase2)
            

    
class simplex_phase2:
    def otimization(self,c,A,b):
        self.objcoefficcient = c
        self.constraintMatrix = A
        self.rightVector = b
        
        #第1段階を解く
        solphase1 = simplex_phase1(self.constraintMatrix,self.rightVector)
        # print(solphase1.phase1Function())
        if  solphase1.phase1Function() is None:
            return "this problem is infeasible"
        else:
            #初期添字集合取得
            initBaseIdx = solphase1.phase1Function()[0]
            initNonbaseIdx = solphase1.phase1Function()[1]
            print(initBaseIdx)
            print(initNonbaseIdx)
            #第2段階を解く
            solphase2 = simplex()
            sol = solphase2.simplexFunction(self.constraintMatrix,self.objcoefficcient,self.rightVector,False,True,initBaseIdx,initNonbaseIdx)
            return sol

# dimx = random.randint(5,10)
# dimy = random.randint(10,19)
# A = np.random.randint(0,10,size=(dimx,dimy))
# c = np.random.randint(0,10,size=(dimy,1))
# b = np.random.randint(0,10,size=(dimx,1))

# A = np.array([[3, 2, 1, 1, 4, 5, 2, 1, 3, 1,1,0,0,0,0],
#              [1, 4, 3, 2, 2, 1, 3, 4, 1, 5,0,1,0,0,0],
#              [2, 1, 4, 3, 3, 2, 5, 1, 4, 3,0,0,1,0,0],
#              [1, 1, 2, 1, 1, 3, 4, 5, 2, 1,0,0,0,1,0],
#              [4, 3, 2, 5, 1, 2, 1, 3, 1, 2,0,0,0,0,1],])
# b = np.array([[100],
#               [80],
#               [90],
#               [70],
#               [85]])
# c = np.array([[-2],
#               [-3],
#               [-1.5],
#               [-4],
#               [-5],
#               [-3],
#               [-6],
#               [-2],
#               [-7],
#               [-2.5],
#               [0],
#               [0],
#               [0],
#               [0],
#               [0]])

#目的関数の係数ベクトル c（列ベクトル）
# c = np.array([-3, -5, -2, -7, -4, -6, -8, -1, -9, -2]).reshape(-1, 1)

# # 制約行列 A
# A = np.array([
#     [2, 3, 1, 4, 1, 2, 5, 6, 3, 1],
#     [1, 2, 3, 1, 4, 5, 2, 1, 3, 2],
#     [3, 1, 2, 5, 1, 1, 6, 2, 4, 3],
#     [4, 3, 1, 1, 5, 2, 1, 3, 2, 4],
#     [1, 1, 2, 3, 4, 1, 5, 2, 1, 2]
# ])

# # 等式制約の右辺ベクトル b（列ベクトル）
b = np.array([50, 40, 60, 70, 55]).reshape(-1, 1)
dimx = random.randint(100,120)
dimy = random.randint(400,500)
A = np.random.randint(0,100,size=(dimx,dimy-dimx))
I = np.identity(dimx)
matrix = np.concatenate((A,I),axis=1)
c = np.random.randint(0,10,size=(dimy-dimx,1))
b = np.random.randint(1,2,size=(dimx,1))
import scipy 
result1 = scipy.optimize.linprog(c,A_eq=A,b_eq=b)
print(result1.message)
print(result1.success)
print(result1.fun)
print(result1.x)
print(result1.nit)
result = simplex_phase2().otimization(c,A,b)
print(result)
print(result1.message)
print(result1.fun)
print(np.array(result1.x).reshape(-1,1))
