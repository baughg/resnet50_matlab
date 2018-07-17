# Resnet50
This is a MATLAB implementation of the [ResNet-50](https://dgschwend.github.io/netscope/#/preset/resnet-50) inference CNN. 
The script 'rn_forward.m' does a forward pass of this network. By default the input image is an African bush elephant and
the script output is shown below:

```
***************************************
1. 0.756  386: 'African elephant, Loxodonta africana',
2. 0.226  101: 'tusker',
3. 0.018  385: 'Indian elephant, Elephas maximus',
4. 0.000  51: 'triceratops',
5. 0.000  978: 'seashore, coast, seacoast, sea-coast',
```


