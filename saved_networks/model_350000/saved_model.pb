ў∙
х╡
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718шо
Т
q_network/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameq_network/conv2d/kernel
Л
+q_network/conv2d/kernel/Read/ReadVariableOpReadVariableOpq_network/conv2d/kernel*&
_output_shapes
: *
dtype0
В
q_network/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameq_network/conv2d/bias
{
)q_network/conv2d/bias/Read/ReadVariableOpReadVariableOpq_network/conv2d/bias*
_output_shapes
: *
dtype0
Ц
q_network/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameq_network/conv2d_1/kernel
П
-q_network/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpq_network/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
Ж
q_network/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameq_network/conv2d_1/bias

+q_network/conv2d_1/bias/Read/ReadVariableOpReadVariableOpq_network/conv2d_1/bias*
_output_shapes
:@*
dtype0
Ц
q_network/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_nameq_network/conv2d_2/kernel
П
-q_network/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpq_network/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0
Ж
q_network/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameq_network/conv2d_2/bias

+q_network/conv2d_2/bias/Read/ReadVariableOpReadVariableOpq_network/conv2d_2/bias*
_output_shapes
:@*
dtype0
К
q_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└А*'
shared_nameq_network/dense/kernel
Г
*q_network/dense/kernel/Read/ReadVariableOpReadVariableOpq_network/dense/kernel* 
_output_shapes
:
└А*
dtype0
Б
q_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameq_network/dense/bias
z
(q_network/dense/bias/Read/ReadVariableOpReadVariableOpq_network/dense/bias*
_output_shapes	
:А*
dtype0
Н
q_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*)
shared_nameq_network/dense_1/kernel
Ж
,q_network/dense_1/kernel/Read/ReadVariableOpReadVariableOpq_network/dense_1/kernel*
_output_shapes
:	А*
dtype0
Д
q_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameq_network/dense_1/bias
}
*q_network/dense_1/bias/Read/ReadVariableOpReadVariableOpq_network/dense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
а
Adam/q_network/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/q_network/conv2d/kernel/m
Щ
2Adam/q_network/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/q_network/conv2d/kernel/m*&
_output_shapes
: *
dtype0
Р
Adam/q_network/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/q_network/conv2d/bias/m
Й
0Adam/q_network/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/q_network/conv2d/bias/m*
_output_shapes
: *
dtype0
д
 Adam/q_network/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*1
shared_name" Adam/q_network/conv2d_1/kernel/m
Э
4Adam/q_network/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/q_network/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0
Ф
Adam/q_network/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/q_network/conv2d_1/bias/m
Н
2Adam/q_network/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/q_network/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
д
 Adam/q_network/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*1
shared_name" Adam/q_network/conv2d_2/kernel/m
Э
4Adam/q_network/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/q_network/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0
Ф
Adam/q_network/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/q_network/conv2d_2/bias/m
Н
2Adam/q_network/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/q_network/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
Ш
Adam/q_network/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└А*.
shared_nameAdam/q_network/dense/kernel/m
С
1Adam/q_network/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/q_network/dense/kernel/m* 
_output_shapes
:
└А*
dtype0
П
Adam/q_network/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_nameAdam/q_network/dense/bias/m
И
/Adam/q_network/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/q_network/dense/bias/m*
_output_shapes	
:А*
dtype0
Ы
Adam/q_network/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*0
shared_name!Adam/q_network/dense_1/kernel/m
Ф
3Adam/q_network/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/q_network/dense_1/kernel/m*
_output_shapes
:	А*
dtype0
Т
Adam/q_network/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/q_network/dense_1/bias/m
Л
1Adam/q_network/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/q_network/dense_1/bias/m*
_output_shapes
:*
dtype0
а
Adam/q_network/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/q_network/conv2d/kernel/v
Щ
2Adam/q_network/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/q_network/conv2d/kernel/v*&
_output_shapes
: *
dtype0
Р
Adam/q_network/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/q_network/conv2d/bias/v
Й
0Adam/q_network/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/q_network/conv2d/bias/v*
_output_shapes
: *
dtype0
д
 Adam/q_network/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*1
shared_name" Adam/q_network/conv2d_1/kernel/v
Э
4Adam/q_network/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/q_network/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0
Ф
Adam/q_network/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/q_network/conv2d_1/bias/v
Н
2Adam/q_network/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/q_network/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
д
 Adam/q_network/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*1
shared_name" Adam/q_network/conv2d_2/kernel/v
Э
4Adam/q_network/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/q_network/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0
Ф
Adam/q_network/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/q_network/conv2d_2/bias/v
Н
2Adam/q_network/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/q_network/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
Ш
Adam/q_network/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└А*.
shared_nameAdam/q_network/dense/kernel/v
С
1Adam/q_network/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/q_network/dense/kernel/v* 
_output_shapes
:
└А*
dtype0
П
Adam/q_network/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_nameAdam/q_network/dense/bias/v
И
/Adam/q_network/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/q_network/dense/bias/v*
_output_shapes	
:А*
dtype0
Ы
Adam/q_network/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*0
shared_name!Adam/q_network/dense_1/kernel/v
Ф
3Adam/q_network/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/q_network/dense_1/kernel/v*
_output_shapes
:	А*
dtype0
Т
Adam/q_network/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/q_network/dense_1/bias/v
Л
1Adam/q_network/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/q_network/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
╢4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ё3
valueч3Bф3 B▌3
╝
	conv1
	pool1
	conv2
	conv3
flatten
fc1
fc2
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
Ї
4iter

5beta_1

6beta_2
	7decay
8learning_ratemambmcmdmemf(mg)mh.mi/mjvkvlvmvnvovp(vq)vr.vs/vt
F
0
1
2
3
4
5
(6
)7
.8
/9
F
0
1
2
3
4
5
(6
)7
.8
/9
 
н
		variables
9non_trainable_variables
:layer_metrics

trainable_variables
;layer_regularization_losses

<layers
regularization_losses
=metrics
 
TR
VARIABLE_VALUEq_network/conv2d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEq_network/conv2d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
	variables
>non_trainable_variables
?layer_metrics
trainable_variables
@layer_regularization_losses

Alayers
regularization_losses
Bmetrics
 
 
 
н
	variables
Cnon_trainable_variables
Dlayer_metrics
trainable_variables
Elayer_regularization_losses

Flayers
regularization_losses
Gmetrics
VT
VARIABLE_VALUEq_network/conv2d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEq_network/conv2d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
	variables
Hnon_trainable_variables
Ilayer_metrics
trainable_variables
Jlayer_regularization_losses

Klayers
regularization_losses
Lmetrics
VT
VARIABLE_VALUEq_network/conv2d_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEq_network/conv2d_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
 	variables
Mnon_trainable_variables
Nlayer_metrics
!trainable_variables
Olayer_regularization_losses

Players
"regularization_losses
Qmetrics
 
 
 
н
$	variables
Rnon_trainable_variables
Slayer_metrics
%trainable_variables
Tlayer_regularization_losses

Ulayers
&regularization_losses
Vmetrics
QO
VARIABLE_VALUEq_network/dense/kernel%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEq_network/dense/bias#fc1/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
н
*	variables
Wnon_trainable_variables
Xlayer_metrics
+trainable_variables
Ylayer_regularization_losses

Zlayers
,regularization_losses
[metrics
SQ
VARIABLE_VALUEq_network/dense_1/kernel%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEq_network/dense_1/bias#fc2/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
н
0	variables
\non_trainable_variables
]layer_metrics
1trainable_variables
^layer_regularization_losses

_layers
2regularization_losses
`metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
wu
VARIABLE_VALUEAdam/q_network/conv2d/kernel/mCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/q_network/conv2d/bias/mAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/q_network/conv2d_1/kernel/mCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/q_network/conv2d_1/bias/mAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/q_network/conv2d_2/kernel/mCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/q_network/conv2d_2/bias/mAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/q_network/dense/kernel/mAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/q_network/dense/bias/m?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/q_network/dense_1/kernel/mAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/q_network/dense_1/bias/m?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/q_network/conv2d/kernel/vCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/q_network/conv2d/bias/vAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/q_network/conv2d_1/kernel/vCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/q_network/conv2d_1/bias/vAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/q_network/conv2d_2/kernel/vCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/q_network/conv2d_2/bias/vAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/q_network/dense/kernel/vAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/q_network/dense/bias/v?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/q_network/dense_1/kernel/vAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/q_network/dense_1/bias/v?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:         PP*
dtype0*$
shape:         PP
├
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1q_network/conv2d/kernelq_network/conv2d/biasq_network/conv2d_1/kernelq_network/conv2d_1/biasq_network/conv2d_2/kernelq_network/conv2d_2/biasq_network/dense/kernelq_network/dense/biasq_network/dense_1/kernelq_network/dense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В */
f*R(
&__inference_signature_wrapper_97146939
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╕
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+q_network/conv2d/kernel/Read/ReadVariableOp)q_network/conv2d/bias/Read/ReadVariableOp-q_network/conv2d_1/kernel/Read/ReadVariableOp+q_network/conv2d_1/bias/Read/ReadVariableOp-q_network/conv2d_2/kernel/Read/ReadVariableOp+q_network/conv2d_2/bias/Read/ReadVariableOp*q_network/dense/kernel/Read/ReadVariableOp(q_network/dense/bias/Read/ReadVariableOp,q_network/dense_1/kernel/Read/ReadVariableOp*q_network/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp2Adam/q_network/conv2d/kernel/m/Read/ReadVariableOp0Adam/q_network/conv2d/bias/m/Read/ReadVariableOp4Adam/q_network/conv2d_1/kernel/m/Read/ReadVariableOp2Adam/q_network/conv2d_1/bias/m/Read/ReadVariableOp4Adam/q_network/conv2d_2/kernel/m/Read/ReadVariableOp2Adam/q_network/conv2d_2/bias/m/Read/ReadVariableOp1Adam/q_network/dense/kernel/m/Read/ReadVariableOp/Adam/q_network/dense/bias/m/Read/ReadVariableOp3Adam/q_network/dense_1/kernel/m/Read/ReadVariableOp1Adam/q_network/dense_1/bias/m/Read/ReadVariableOp2Adam/q_network/conv2d/kernel/v/Read/ReadVariableOp0Adam/q_network/conv2d/bias/v/Read/ReadVariableOp4Adam/q_network/conv2d_1/kernel/v/Read/ReadVariableOp2Adam/q_network/conv2d_1/bias/v/Read/ReadVariableOp4Adam/q_network/conv2d_2/kernel/v/Read/ReadVariableOp2Adam/q_network/conv2d_2/bias/v/Read/ReadVariableOp1Adam/q_network/dense/kernel/v/Read/ReadVariableOp/Adam/q_network/dense/bias/v/Read/ReadVariableOp3Adam/q_network/dense_1/kernel/v/Read/ReadVariableOp1Adam/q_network/dense_1/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_save_97147178
ў	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameq_network/conv2d/kernelq_network/conv2d/biasq_network/conv2d_1/kernelq_network/conv2d_1/biasq_network/conv2d_2/kernelq_network/conv2d_2/biasq_network/dense/kernelq_network/dense/biasq_network/dense_1/kernelq_network/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/q_network/conv2d/kernel/mAdam/q_network/conv2d/bias/m Adam/q_network/conv2d_1/kernel/mAdam/q_network/conv2d_1/bias/m Adam/q_network/conv2d_2/kernel/mAdam/q_network/conv2d_2/bias/mAdam/q_network/dense/kernel/mAdam/q_network/dense/bias/mAdam/q_network/dense_1/kernel/mAdam/q_network/dense_1/bias/mAdam/q_network/conv2d/kernel/vAdam/q_network/conv2d/bias/v Adam/q_network/conv2d_1/kernel/vAdam/q_network/conv2d_1/bias/v Adam/q_network/conv2d_2/kernel/vAdam/q_network/conv2d_2/bias/vAdam/q_network/dense/kernel/vAdam/q_network/dense/bias/vAdam/q_network/dense_1/kernel/vAdam/q_network/dense_1/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference__traced_restore_97147293ЫН
│

ў
E__inference_dense_1_layer_call_and_return_conditional_losses_97147050

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╩
а
+__inference_conv2d_1_layer_call_fn_97146968

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_971467582
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         

 
 
_user_specified_nameinputs
╩@
Э	
#__inference__wrapped_model_97146713
input_1I
/q_network_conv2d_conv2d_readvariableop_resource: >
0q_network_conv2d_biasadd_readvariableop_resource: K
1q_network_conv2d_1_conv2d_readvariableop_resource: @@
2q_network_conv2d_1_biasadd_readvariableop_resource:@K
1q_network_conv2d_2_conv2d_readvariableop_resource:@@@
2q_network_conv2d_2_biasadd_readvariableop_resource:@B
.q_network_dense_matmul_readvariableop_resource:
└А>
/q_network_dense_biasadd_readvariableop_resource:	АC
0q_network_dense_1_matmul_readvariableop_resource:	А?
1q_network_dense_1_biasadd_readvariableop_resource:
identityИв'q_network/conv2d/BiasAdd/ReadVariableOpв&q_network/conv2d/Conv2D/ReadVariableOpв)q_network/conv2d_1/BiasAdd/ReadVariableOpв(q_network/conv2d_1/Conv2D/ReadVariableOpв)q_network/conv2d_2/BiasAdd/ReadVariableOpв(q_network/conv2d_2/Conv2D/ReadVariableOpв&q_network/dense/BiasAdd/ReadVariableOpв%q_network/dense/MatMul/ReadVariableOpв(q_network/dense_1/BiasAdd/ReadVariableOpв'q_network/dense_1/MatMul/ReadVariableOp╚
&q_network/conv2d/Conv2D/ReadVariableOpReadVariableOp/q_network_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&q_network/conv2d/Conv2D/ReadVariableOp╫
q_network/conv2d/Conv2DConv2Dinput_1.q_network/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
q_network/conv2d/Conv2D┐
'q_network/conv2d/BiasAdd/ReadVariableOpReadVariableOp0q_network_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'q_network/conv2d/BiasAdd/ReadVariableOp╠
q_network/conv2d/BiasAddBiasAdd q_network/conv2d/Conv2D:output:0/q_network/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
q_network/conv2d/BiasAddУ
q_network/conv2d/ReluRelu!q_network/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:          2
q_network/conv2d/Relu▐
q_network/max_pooling2d/MaxPoolMaxPool#q_network/conv2d/Relu:activations:0*/
_output_shapes
:         

 *
ksize
*
paddingSAME*
strides
2!
q_network/max_pooling2d/MaxPool╬
(q_network/conv2d_1/Conv2D/ReadVariableOpReadVariableOp1q_network_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02*
(q_network/conv2d_1/Conv2D/ReadVariableOp■
q_network/conv2d_1/Conv2DConv2D(q_network/max_pooling2d/MaxPool:output:00q_network/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
q_network/conv2d_1/Conv2D┼
)q_network/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2q_network_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)q_network/conv2d_1/BiasAdd/ReadVariableOp╘
q_network/conv2d_1/BiasAddBiasAdd"q_network/conv2d_1/Conv2D:output:01q_network/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
q_network/conv2d_1/BiasAddЩ
q_network/conv2d_1/ReluRelu#q_network/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
q_network/conv2d_1/Relu╬
(q_network/conv2d_2/Conv2D/ReadVariableOpReadVariableOp1q_network_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(q_network/conv2d_2/Conv2D/ReadVariableOp√
q_network/conv2d_2/Conv2DConv2D%q_network/conv2d_1/Relu:activations:00q_network/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
q_network/conv2d_2/Conv2D┼
)q_network/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2q_network_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)q_network/conv2d_2/BiasAdd/ReadVariableOp╘
q_network/conv2d_2/BiasAddBiasAdd"q_network/conv2d_2/Conv2D:output:01q_network/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
q_network/conv2d_2/BiasAddЩ
q_network/conv2d_2/ReluRelu#q_network/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
q_network/conv2d_2/ReluГ
q_network/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
q_network/flatten/Const╜
q_network/flatten/ReshapeReshape%q_network/conv2d_2/Relu:activations:0 q_network/flatten/Const:output:0*
T0*(
_output_shapes
:         └2
q_network/flatten/Reshape┐
%q_network/dense/MatMul/ReadVariableOpReadVariableOp.q_network_dense_matmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype02'
%q_network/dense/MatMul/ReadVariableOp└
q_network/dense/MatMulMatMul"q_network/flatten/Reshape:output:0-q_network/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
q_network/dense/MatMul╜
&q_network/dense/BiasAdd/ReadVariableOpReadVariableOp/q_network_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02(
&q_network/dense/BiasAdd/ReadVariableOp┬
q_network/dense/BiasAddBiasAdd q_network/dense/MatMul:product:0.q_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
q_network/dense/BiasAddЙ
q_network/dense/ReluRelu q_network/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
q_network/dense/Relu─
'q_network/dense_1/MatMul/ReadVariableOpReadVariableOp0q_network_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'q_network/dense_1/MatMul/ReadVariableOp┼
q_network/dense_1/MatMulMatMul"q_network/dense/Relu:activations:0/q_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
q_network/dense_1/MatMul┬
(q_network/dense_1/BiasAdd/ReadVariableOpReadVariableOp1q_network_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(q_network/dense_1/BiasAdd/ReadVariableOp╔
q_network/dense_1/BiasAddBiasAdd"q_network/dense_1/MatMul:product:00q_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
q_network/dense_1/BiasAddЧ
q_network/dense_1/SigmoidSigmoid"q_network/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
q_network/dense_1/SigmoidШ
IdentityIdentityq_network/dense_1/Sigmoid:y:0(^q_network/conv2d/BiasAdd/ReadVariableOp'^q_network/conv2d/Conv2D/ReadVariableOp*^q_network/conv2d_1/BiasAdd/ReadVariableOp)^q_network/conv2d_1/Conv2D/ReadVariableOp*^q_network/conv2d_2/BiasAdd/ReadVariableOp)^q_network/conv2d_2/Conv2D/ReadVariableOp'^q_network/dense/BiasAdd/ReadVariableOp&^q_network/dense/MatMul/ReadVariableOp)^q_network/dense_1/BiasAdd/ReadVariableOp(^q_network/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         PP: : : : : : : : : : 2R
'q_network/conv2d/BiasAdd/ReadVariableOp'q_network/conv2d/BiasAdd/ReadVariableOp2P
&q_network/conv2d/Conv2D/ReadVariableOp&q_network/conv2d/Conv2D/ReadVariableOp2V
)q_network/conv2d_1/BiasAdd/ReadVariableOp)q_network/conv2d_1/BiasAdd/ReadVariableOp2T
(q_network/conv2d_1/Conv2D/ReadVariableOp(q_network/conv2d_1/Conv2D/ReadVariableOp2V
)q_network/conv2d_2/BiasAdd/ReadVariableOp)q_network/conv2d_2/BiasAdd/ReadVariableOp2T
(q_network/conv2d_2/Conv2D/ReadVariableOp(q_network/conv2d_2/Conv2D/ReadVariableOp2P
&q_network/dense/BiasAdd/ReadVariableOp&q_network/dense/BiasAdd/ReadVariableOp2N
%q_network/dense/MatMul/ReadVariableOp%q_network/dense/MatMul/ReadVariableOp2T
(q_network/dense_1/BiasAdd/ReadVariableOp(q_network/dense_1/BiasAdd/ReadVariableOp2R
'q_network/dense_1/MatMul/ReadVariableOp'q_network/dense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:         PP
!
_user_specified_name	input_1
╖

ў
C__inference_dense_layer_call_and_return_conditional_losses_97147030

inputs2
matmul_readvariableop_resource:
└А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╙
F
*__inference_flatten_layer_call_fn_97147004

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_971467872
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
г

О
,__inference_q_network_layer_call_fn_97146850
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
└А
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_q_network_layer_call_and_return_conditional_losses_971468242
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         PP: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         PP
!
_user_specified_name	input_1
Ц
 
F__inference_conv2d_2_layer_call_and_return_conditional_losses_97146775

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
к
g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_97146719

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╩
а
+__inference_conv2d_2_layer_call_fn_97146988

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_971467752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
М#
■
G__inference_q_network_layer_call_and_return_conditional_losses_97146824
input_1)
conv2d_97146741: 
conv2d_97146743: +
conv2d_1_97146759: @
conv2d_1_97146761:@+
conv2d_2_97146776:@@
conv2d_2_97146778:@"
dense_97146801:
└А
dense_97146803:	А#
dense_1_97146818:	А
dense_1_97146820:
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallЬ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_97146741conv2d_97146743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_971467402 
conv2d/StatefulPartitionedCallС
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_971467192
max_pooling2d/PartitionedCall┼
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_97146759conv2d_1_97146761*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_971467582"
 conv2d_1/StatefulPartitionedCall╚
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_97146776conv2d_2_97146778*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_971467752"
 conv2d_2/StatefulPartitionedCall·
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_971467872
flatten/PartitionedCallй
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_97146801dense_97146803*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_971468002
dense/StatefulPartitionedCall╕
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_97146818dense_1_97146820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_971468172!
dense_1/StatefulPartitionedCallе
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         PP: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
/
_output_shapes
:         PP
!
_user_specified_name	input_1
ч
a
E__inference_flatten_layer_call_and_return_conditional_losses_97146787

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
│

ў
E__inference_dense_1_layer_call_and_return_conditional_losses_97146817

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ф
¤
D__inference_conv2d_layer_call_and_return_conditional_losses_97146740

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         PP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         PP
 
_user_specified_nameinputs
г
Ш
*__inference_dense_1_layer_call_fn_97147039

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_971468172
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ф
¤
D__inference_conv2d_layer_call_and_return_conditional_losses_97146959

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         PP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         PP
 
_user_specified_nameinputs
г
Ш
(__inference_dense_layer_call_fn_97147019

inputs
unknown:
└А
	unknown_0:	А
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_971468002
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
┌
L
0__inference_max_pooling2d_layer_call_fn_97146725

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_971467192
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
─N
ф
!__inference__traced_save_97147178
file_prefix6
2savev2_q_network_conv2d_kernel_read_readvariableop4
0savev2_q_network_conv2d_bias_read_readvariableop8
4savev2_q_network_conv2d_1_kernel_read_readvariableop6
2savev2_q_network_conv2d_1_bias_read_readvariableop8
4savev2_q_network_conv2d_2_kernel_read_readvariableop6
2savev2_q_network_conv2d_2_bias_read_readvariableop5
1savev2_q_network_dense_kernel_read_readvariableop3
/savev2_q_network_dense_bias_read_readvariableop7
3savev2_q_network_dense_1_kernel_read_readvariableop5
1savev2_q_network_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop=
9savev2_adam_q_network_conv2d_kernel_m_read_readvariableop;
7savev2_adam_q_network_conv2d_bias_m_read_readvariableop?
;savev2_adam_q_network_conv2d_1_kernel_m_read_readvariableop=
9savev2_adam_q_network_conv2d_1_bias_m_read_readvariableop?
;savev2_adam_q_network_conv2d_2_kernel_m_read_readvariableop=
9savev2_adam_q_network_conv2d_2_bias_m_read_readvariableop<
8savev2_adam_q_network_dense_kernel_m_read_readvariableop:
6savev2_adam_q_network_dense_bias_m_read_readvariableop>
:savev2_adam_q_network_dense_1_kernel_m_read_readvariableop<
8savev2_adam_q_network_dense_1_bias_m_read_readvariableop=
9savev2_adam_q_network_conv2d_kernel_v_read_readvariableop;
7savev2_adam_q_network_conv2d_bias_v_read_readvariableop?
;savev2_adam_q_network_conv2d_1_kernel_v_read_readvariableop=
9savev2_adam_q_network_conv2d_1_bias_v_read_readvariableop?
;savev2_adam_q_network_conv2d_2_kernel_v_read_readvariableop=
9savev2_adam_q_network_conv2d_2_bias_v_read_readvariableop<
8savev2_adam_q_network_dense_kernel_v_read_readvariableop:
6savev2_adam_q_network_dense_bias_v_read_readvariableop>
:savev2_adam_q_network_dense_1_kernel_v_read_readvariableop<
8savev2_adam_q_network_dense_1_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╬
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*р
value╓B╙$B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╨
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╟
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_q_network_conv2d_kernel_read_readvariableop0savev2_q_network_conv2d_bias_read_readvariableop4savev2_q_network_conv2d_1_kernel_read_readvariableop2savev2_q_network_conv2d_1_bias_read_readvariableop4savev2_q_network_conv2d_2_kernel_read_readvariableop2savev2_q_network_conv2d_2_bias_read_readvariableop1savev2_q_network_dense_kernel_read_readvariableop/savev2_q_network_dense_bias_read_readvariableop3savev2_q_network_dense_1_kernel_read_readvariableop1savev2_q_network_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop9savev2_adam_q_network_conv2d_kernel_m_read_readvariableop7savev2_adam_q_network_conv2d_bias_m_read_readvariableop;savev2_adam_q_network_conv2d_1_kernel_m_read_readvariableop9savev2_adam_q_network_conv2d_1_bias_m_read_readvariableop;savev2_adam_q_network_conv2d_2_kernel_m_read_readvariableop9savev2_adam_q_network_conv2d_2_bias_m_read_readvariableop8savev2_adam_q_network_dense_kernel_m_read_readvariableop6savev2_adam_q_network_dense_bias_m_read_readvariableop:savev2_adam_q_network_dense_1_kernel_m_read_readvariableop8savev2_adam_q_network_dense_1_bias_m_read_readvariableop9savev2_adam_q_network_conv2d_kernel_v_read_readvariableop7savev2_adam_q_network_conv2d_bias_v_read_readvariableop;savev2_adam_q_network_conv2d_1_kernel_v_read_readvariableop9savev2_adam_q_network_conv2d_1_bias_v_read_readvariableop;savev2_adam_q_network_conv2d_2_kernel_v_read_readvariableop9savev2_adam_q_network_conv2d_2_bias_v_read_readvariableop8savev2_adam_q_network_dense_kernel_v_read_readvariableop6savev2_adam_q_network_dense_bias_v_read_readvariableop:savev2_adam_q_network_dense_1_kernel_v_read_readvariableop8savev2_adam_q_network_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ч
_input_shapes╒
╥: : : : @:@:@@:@:
└А:А:	А:: : : : : : : : @:@:@@:@:
└А:А:	А:: : : @:@:@@:@:
└А:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
└А:!

_output_shapes	
:А:%	!

_output_shapes
:	А: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
└А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:& "
 
_output_shapes
:
└А:!!

_output_shapes	
:А:%"!

_output_shapes
:	А: #

_output_shapes
::$

_output_shapes
: 
∙	
И
&__inference_signature_wrapper_97146939
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
└А
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__wrapped_model_971467132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         PP: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         PP
!
_user_specified_name	input_1
Ц
 
F__inference_conv2d_2_layer_call_and_return_conditional_losses_97146999

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
¤Ш
Р
$__inference__traced_restore_97147293
file_prefixB
(assignvariableop_q_network_conv2d_kernel: 6
(assignvariableop_1_q_network_conv2d_bias: F
,assignvariableop_2_q_network_conv2d_1_kernel: @8
*assignvariableop_3_q_network_conv2d_1_bias:@F
,assignvariableop_4_q_network_conv2d_2_kernel:@@8
*assignvariableop_5_q_network_conv2d_2_bias:@=
)assignvariableop_6_q_network_dense_kernel:
└А6
'assignvariableop_7_q_network_dense_bias:	А>
+assignvariableop_8_q_network_dense_1_kernel:	А7
)assignvariableop_9_q_network_dense_1_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: L
2assignvariableop_15_adam_q_network_conv2d_kernel_m: >
0assignvariableop_16_adam_q_network_conv2d_bias_m: N
4assignvariableop_17_adam_q_network_conv2d_1_kernel_m: @@
2assignvariableop_18_adam_q_network_conv2d_1_bias_m:@N
4assignvariableop_19_adam_q_network_conv2d_2_kernel_m:@@@
2assignvariableop_20_adam_q_network_conv2d_2_bias_m:@E
1assignvariableop_21_adam_q_network_dense_kernel_m:
└А>
/assignvariableop_22_adam_q_network_dense_bias_m:	АF
3assignvariableop_23_adam_q_network_dense_1_kernel_m:	А?
1assignvariableop_24_adam_q_network_dense_1_bias_m:L
2assignvariableop_25_adam_q_network_conv2d_kernel_v: >
0assignvariableop_26_adam_q_network_conv2d_bias_v: N
4assignvariableop_27_adam_q_network_conv2d_1_kernel_v: @@
2assignvariableop_28_adam_q_network_conv2d_1_bias_v:@N
4assignvariableop_29_adam_q_network_conv2d_2_kernel_v:@@@
2assignvariableop_30_adam_q_network_conv2d_2_bias_v:@E
1assignvariableop_31_adam_q_network_dense_kernel_v:
└А>
/assignvariableop_32_adam_q_network_dense_bias_v:	АF
3assignvariableop_33_adam_q_network_dense_1_kernel_v:	А?
1assignvariableop_34_adam_q_network_dense_1_bias_v:
identity_36ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╘
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*р
value╓B╙$B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc1/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc1/bias/.ATTRIBUTES/VARIABLE_VALUEB%fc2/kernel/.ATTRIBUTES/VARIABLE_VALUEB#fc2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAfc2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?fc2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╓
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesт
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ж
_output_shapesУ
Р::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityз
AssignVariableOpAssignVariableOp(assignvariableop_q_network_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1н
AssignVariableOp_1AssignVariableOp(assignvariableop_1_q_network_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2▒
AssignVariableOp_2AssignVariableOp,assignvariableop_2_q_network_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3п
AssignVariableOp_3AssignVariableOp*assignvariableop_3_q_network_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_q_network_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5п
AssignVariableOp_5AssignVariableOp*assignvariableop_5_q_network_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6о
AssignVariableOp_6AssignVariableOp)assignvariableop_6_q_network_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7м
AssignVariableOp_7AssignVariableOp'assignvariableop_7_q_network_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8░
AssignVariableOp_8AssignVariableOp+assignvariableop_8_q_network_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9о
AssignVariableOp_9AssignVariableOp)assignvariableop_9_q_network_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10е
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11з
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12з
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ж
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14о
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15║
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adam_q_network_conv2d_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╕
AssignVariableOp_16AssignVariableOp0assignvariableop_16_adam_q_network_conv2d_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╝
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_q_network_conv2d_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18║
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_q_network_conv2d_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╝
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_q_network_conv2d_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20║
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_q_network_conv2d_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╣
AssignVariableOp_21AssignVariableOp1assignvariableop_21_adam_q_network_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╖
AssignVariableOp_22AssignVariableOp/assignvariableop_22_adam_q_network_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╗
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_q_network_dense_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╣
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_q_network_dense_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25║
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_q_network_conv2d_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╕
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adam_q_network_conv2d_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╝
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_q_network_conv2d_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28║
AssignVariableOp_28AssignVariableOp2assignvariableop_28_adam_q_network_conv2d_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╝
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_q_network_conv2d_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30║
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_q_network_conv2d_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╣
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_q_network_dense_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╖
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_q_network_dense_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╗
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_q_network_dense_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╣
AssignVariableOp_34AssignVariableOp1assignvariableop_34_adam_q_network_dense_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpр
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35╙
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ц
 
F__inference_conv2d_1_layer_call_and_return_conditional_losses_97146979

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         

 
 
_user_specified_nameinputs
╖

ў
C__inference_dense_layer_call_and_return_conditional_losses_97146800

inputs2
matmul_readvariableop_resource:
└А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╞
Ю
)__inference_conv2d_layer_call_fn_97146948

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_971467402
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         PP: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         PP
 
_user_specified_nameinputs
Ц
 
F__inference_conv2d_1_layer_call_and_return_conditional_losses_97146758

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         

 
 
_user_specified_nameinputs
ч
a
E__inference_flatten_layer_call_and_return_conditional_losses_97147010

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
C
input_18
serving_default_input_1:0         PP<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:║╡
Ю

	conv1
	pool1
	conv2
	conv3
flatten
fc1
fc2
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
u_default_save_signature
v__call__
*w&call_and_return_all_conditional_losses"И
_tf_keras_modelю{"name": "q_network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 80, 80, 4]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "QNetwork"}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999974752427e-07, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
╚


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"г	
_tf_keras_layerЙ	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}, "shared_object_id": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 80, 4]}}
и
	variables
trainable_variables
regularization_losses
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"Щ
_tf_keras_layer {"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 6}}
╧


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
|__call__
*}&call_and_return_all_conditional_losses"к	
_tf_keras_layerР	{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 32]}}
╨


kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
~__call__
*&call_and_return_all_conditional_losses"л	
_tf_keras_layerС	{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 64]}}
Ф
$	variables
%trainable_variables
&regularization_losses
'	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"Г
_tf_keras_layerщ{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 16}}
╙

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"м
_tf_keras_layerТ{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1600}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1600]}}
╓

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"п
_tf_keras_layerХ{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
З
4iter

5beta_1

6beta_2
	7decay
8learning_ratemambmcmdmemf(mg)mh.mi/mjvkvlvmvnvovp(vq)vr.vs/vt"
	optimizer
f
0
1
2
3
4
5
(6
)7
.8
/9"
trackable_list_wrapper
f
0
1
2
3
4
5
(6
)7
.8
/9"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
		variables
9non_trainable_variables
:layer_metrics

trainable_variables
;layer_regularization_losses

<layers
regularization_losses
=metrics
v__call__
u_default_save_signature
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
-
Жserving_default"
signature_map
1:/ 2q_network/conv2d/kernel
#:! 2q_network/conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
	variables
>non_trainable_variables
?layer_metrics
trainable_variables
@layer_regularization_losses

Alayers
regularization_losses
Bmetrics
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
	variables
Cnon_trainable_variables
Dlayer_metrics
trainable_variables
Elayer_regularization_losses

Flayers
regularization_losses
Gmetrics
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
3:1 @2q_network/conv2d_1/kernel
%:#@2q_network/conv2d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
	variables
Hnon_trainable_variables
Ilayer_metrics
trainable_variables
Jlayer_regularization_losses

Klayers
regularization_losses
Lmetrics
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
3:1@@2q_network/conv2d_2/kernel
%:#@2q_network/conv2d_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
 	variables
Mnon_trainable_variables
Nlayer_metrics
!trainable_variables
Olayer_regularization_losses

Players
"regularization_losses
Qmetrics
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
$	variables
Rnon_trainable_variables
Slayer_metrics
%trainable_variables
Tlayer_regularization_losses

Ulayers
&regularization_losses
Vmetrics
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
*:(
└А2q_network/dense/kernel
#:!А2q_network/dense/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
*	variables
Wnon_trainable_variables
Xlayer_metrics
+trainable_variables
Ylayer_regularization_losses

Zlayers
,regularization_losses
[metrics
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
+:)	А2q_network/dense_1/kernel
$:"2q_network/dense_1/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
0	variables
\non_trainable_variables
]layer_metrics
1trainable_variables
^layer_regularization_losses

_layers
2regularization_losses
`metrics
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
6:4 2Adam/q_network/conv2d/kernel/m
(:& 2Adam/q_network/conv2d/bias/m
8:6 @2 Adam/q_network/conv2d_1/kernel/m
*:(@2Adam/q_network/conv2d_1/bias/m
8:6@@2 Adam/q_network/conv2d_2/kernel/m
*:(@2Adam/q_network/conv2d_2/bias/m
/:-
└А2Adam/q_network/dense/kernel/m
(:&А2Adam/q_network/dense/bias/m
0:.	А2Adam/q_network/dense_1/kernel/m
):'2Adam/q_network/dense_1/bias/m
6:4 2Adam/q_network/conv2d/kernel/v
(:& 2Adam/q_network/conv2d/bias/v
8:6 @2 Adam/q_network/conv2d_1/kernel/v
*:(@2Adam/q_network/conv2d_1/bias/v
8:6@@2 Adam/q_network/conv2d_2/kernel/v
*:(@2Adam/q_network/conv2d_2/bias/v
/:-
└А2Adam/q_network/dense/kernel/v
(:&А2Adam/q_network/dense/bias/v
0:.	А2Adam/q_network/dense_1/kernel/v
):'2Adam/q_network/dense_1/bias/v
щ2ц
#__inference__wrapped_model_97146713╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_1         PP
¤2·
,__inference_q_network_layer_call_fn_97146850╔
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_1         PP
Ш2Х
G__inference_q_network_layer_call_and_return_conditional_losses_97146824╔
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_1         PP
╙2╨
)__inference_conv2d_layer_call_fn_97146948в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_conv2d_layer_call_and_return_conditional_losses_97146959в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ш2Х
0__inference_max_pooling2d_layer_call_fn_97146725р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
│2░
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_97146719р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╒2╥
+__inference_conv2d_1_layer_call_fn_97146968в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_conv2d_1_layer_call_and_return_conditional_losses_97146979в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_conv2d_2_layer_call_fn_97146988в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_conv2d_2_layer_call_and_return_conditional_losses_97146999в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_flatten_layer_call_fn_97147004в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_flatten_layer_call_and_return_conditional_losses_97147010в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_layer_call_fn_97147019в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_layer_call_and_return_conditional_losses_97147030в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_1_layer_call_fn_97147039в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_1_layer_call_and_return_conditional_losses_97147050в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
═B╩
&__inference_signature_wrapper_97146939input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 в
#__inference__wrapped_model_97146713{
()./8в5
.в+
)К&
input_1         PP
к "3к0
.
output_1"К
output_1         ╢
F__inference_conv2d_1_layer_call_and_return_conditional_losses_97146979l7в4
-в*
(К%
inputs         

 
к "-в*
#К 
0         @
Ъ О
+__inference_conv2d_1_layer_call_fn_97146968_7в4
-в*
(К%
inputs         

 
к " К         @╢
F__inference_conv2d_2_layer_call_and_return_conditional_losses_97146999l7в4
-в*
(К%
inputs         @
к "-в*
#К 
0         @
Ъ О
+__inference_conv2d_2_layer_call_fn_97146988_7в4
-в*
(К%
inputs         @
к " К         @┤
D__inference_conv2d_layer_call_and_return_conditional_losses_97146959l7в4
-в*
(К%
inputs         PP
к "-в*
#К 
0          
Ъ М
)__inference_conv2d_layer_call_fn_97146948_7в4
-в*
(К%
inputs         PP
к " К          ж
E__inference_dense_1_layer_call_and_return_conditional_losses_97147050]./0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ ~
*__inference_dense_1_layer_call_fn_97147039P./0в-
&в#
!К
inputs         А
к "К         е
C__inference_dense_layer_call_and_return_conditional_losses_97147030^()0в-
&в#
!К
inputs         └
к "&в#
К
0         А
Ъ }
(__inference_dense_layer_call_fn_97147019Q()0в-
&в#
!К
inputs         └
к "К         Ак
E__inference_flatten_layer_call_and_return_conditional_losses_97147010a7в4
-в*
(К%
inputs         @
к "&в#
К
0         └
Ъ В
*__inference_flatten_layer_call_fn_97147004T7в4
-в*
(К%
inputs         @
к "К         └ю
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_97146719ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_layer_call_fn_97146725СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╕
G__inference_q_network_layer_call_and_return_conditional_losses_97146824m
()./8в5
.в+
)К&
input_1         PP
к "%в"
К
0         
Ъ Р
,__inference_q_network_layer_call_fn_97146850`
()./8в5
.в+
)К&
input_1         PP
к "К         ▒
&__inference_signature_wrapper_97146939Ж
()./Cв@
в 
9к6
4
input_1)К&
input_1         PP"3к0
.
output_1"К
output_1         