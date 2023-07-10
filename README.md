# Real_Estate_Prediction
**Project Description** : การใช้ Machine Learning ในการทำนายราคาบ้าน โดยใช้ Simple Linear Regression algorithm & Regularization

<h3>DATA Describtion</h3>
Price = ราคาบ้าน <br>
Bedroom = จำนวนห้องนอน <br>
Space = ขนาดพื้นที่ใช้สอยภายในอาคาร <br>
Room = จำนวนห้องทั้งหมด <br>
Lot = ขนาดของที่ดินที่ตั้งทรัพย์สิน <br>
Tax = ภาษีตามมูลค่าของทรัพย์สิน <br>
Bathroom = จำนวนห้องน้ำ <br>
Garage = จำนวนที่จอดรถ <br>
Condition = เงื่อนไข ( 1 = มี , 0 = ไม่มี ) <br>

# <h3>DATA Pre-processing</h3>

**STEP 1** : Inspec <br>

ก่อนอื่นเราก็จะทำการตรวจสอบว่า ในแต่ละ Column นั้นมีการกระจายตัวเป็นอย่างไร <br>

![003](https://user-images.githubusercontent.com/118663358/235354687-651b39d2-dbf7-4375-a308-024d970a57da.png)

และตรวจสอบดู Outliers

![001](https://user-images.githubusercontent.com/118663358/235354715-0409f4bf-89bc-48af-96a1-d5ea29fe5b7f.png)

เมื่อตรวจสอบแล้วเราก็จะมีตัวเลือกอยู่ 2 อย่างในการจัดการกับ Data คือ <br>
> - เติมค่า NaN ของแต่ละ Column ด้วยการเลือกใช้ Median และ ลบ Outliers ออกไป <br>
> - ลบค่า NaN และ Outliers <br>

ซึ่งเราตัดสินใจใช้ตัวเลือกที่ 2 โดยที่จะยังเก็บ Original Data เอาไว้ เพื่อทำการเปรียบเทียบ <br><br>
จากนั้น <br>
เราก็จะตรวจสอบค่า Correlation ระหว่าง Features เพื่อพิจารณา เพิ่ม-ลด Features ในขั้นตอนถัดไป <br><br>

![002](https://user-images.githubusercontent.com/118663358/235355358-93a2b8e7-4d0e-43d1-919e-b1cc404d062d.png)
<br>

**STEP 2** : Data Preparation <br>

ในขั้นตอนนี้ เราจะทำการแยก X, y <br><br>

```ruby

from sklearn.model_selection import train_test_split

# Original data

x = data.drop( columns='Price' )
y = data['Price']

# Non-original data

x2 = data2.drop( columns='Price' )
y2 = data2['Price']

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.30, random_state = 42 )
X2_train, X2_test, y2_train, y2_test = train_test_split( x2, y2, test_size = 0.30, random_state = 42 )

```
<br>
จากนั้นเราก็จะทำการทดสอบคะแนน ( accuracy score ) ตาม Code ด้านล่าง : <br><br>

```ruby

from sklearn.linear_model import LinearRegression

# Original Data

lr = LinearRegression().fit(X_train,y_train)
print(f'Non Feature Scale & Feature Selection ,also Not remove any outlier : {lr.score(x,y)}')

# Non-original data

lr2 = LinearRegression().fit(X2_train,y2_train)
print(f'Feature Scale but non-Feature Selection & remove outlier : {lr2.score(x2,y2)}')

```

Output : <br>

> Non Feature Scale & Feature Selection ,also Not remove any outlier : 0.7098798840646457 <br>
> Feature Scale but non-Feature Selection & remove outlier : 0.54397304402341 <br>

<br>
เมื่อทำการทดสอบคะแนนเบื้องต้น ก็พบว่า Original Data หรือ Data ที่ไม่ได้ปรับแต่งอะไรเลย มีคะแนนที่สูงกว่า <br><br>
อย่างไรก็ตาม, เราก็เลือกที่จะยังคงปรับแต่ง Data2 ต่อไป โดยการใช้ Pipeline <br><br>
<br>

**STEP 3** : TRAIN <br>

ภายใน Pipeline จะประกอบไปด้วย Standard Scaler , Polynomial Features และ Ridge <br><br>
ทั้งหมดนี้เป็นการ ***เพิ่ม*** ความแม่นยำให้กับ Model หรืออีกนัยหนึ่งก็คือการลดปัญหา 'Underfit' ภายใน Model นั่นเอง <br><br>

```ruby

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# The following codes are intended for use with data2.

alph = 0.1
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha = alph, fit_intercept = True))
]

pipeline = Pipeline(steps)
pr = pipeline.fit(X2_train, y2_train)

print('Training score: {}'.format(pipeline.score(X2_train, y2_train)))
print('Test score: {}'.format(pipeline.score(X2_test, y2_test)))

```

Output : <br>

> Training score: 0.9515430678077388 <br>
> Test score: 0.8656925766049421 <br>

<br>
สังเกตว่าคะแนนที่ได้หลังจากการปรับแต่ง Model แล้ว มีค่าออกมาสูงกว่า Model ของ Original Data มากพอสมควร <br><br>

**STEP 4** : TEST

```ruby

from sklearn.metrics import mean_squared_error, r2_score

y_pred = lr.predict(X_test) # Model from Original data
y2_pred = pr.predict(X2_test)

# The mean squared error
print("Mean squared error ( Original ) : %.2f" % mean_squared_error(y_test,y_pred))
print("Mean squared error : %.2f" % mean_squared_error(y2_test,y2_pred))

# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination ( Original ) : %.2f" % r2_score(y_test,y_pred))
print("Coefficient of determination : %.2f" % r2_score(y2_test,y2_pred))

```

Output : <br>

> Mean squared error ( Original ) : 47.76 <br>
> Mean squared error : 9.97 <br>
> Coefficient of determination ( Original ) : 0.66 <br>
> Coefficient of determination : 0.87 <br>

<br>
จากการทดสอบคะแนนจากการทำนายผลลัพธ์ ตาม Code ด้านบน <br>
จะเห็นได้ว่า Mean squared error ของ Model จาก Data ที่ได้ปรับแต่งไปแล้วมีค่า "น้อยกว่า" Original Model <br><br>

ซึ่งก็หมายความว่า มีความแม่นยำ "มากกว่า" นั่นเอง

# 
<br>
สามารถดู Code โดยละเอียดได้ <a href="https://github.com/HikariJadeEmpire/Real_Estate_Prediction/blob/main/Fit_Linear.ipynb">ที่นี่</a>


