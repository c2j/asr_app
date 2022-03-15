# 生成私钥，按照提示填写内容
openssl genrsa -des3 -out server.key 1024
 
# 生成csr文件 ，按照提示填写内容
openssl req -new -key server.key -out server.csr
 
# Remove Passphrase from key
cp server.key server.key.org 
openssl rsa -in server.key.org -out server.key
 
# 生成crt文件，有效期1年（365天）
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

————————————————
版权声明：本文为CSDN博主「dyingstraw」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/dyingstraw/article/details/82698639