# 生成证书
## 生成私钥，按照提示填写内容
openssl genrsa -des3 -out server.key 1024
 
## 生成csr文件 ，按照提示填写内容
openssl req -new -key server.key -out server.csr
 
## Remove Passphrase from key
cp server.key server.key.org 
openssl rsa -in server.key.org -out server.key
 
## 生成crt文件，有效期1年（365天）
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

# 解决新版 Chrome 中 NET::ERR_CERT_INVALID 不能继续的问题

在 Chrome 中，访问不受信任的 HTTPS 网站时，一般会提示NET::ERR_CERT_INVALID。这可能是由于证书过期、不匹配或者可能用的是自签名证书等情况。通常点击高级按钮会有一个继续按钮，浏览器就会关闭当前页面警告并继续访问。但在高版本的 Chrome for mac 中，是没有这个继续按钮的。

其实，Chrome 中只是隐藏了这个按钮，还是可以通过一个简单的办法来触发继续的功能的。

那就是点击页面的空白处，直接在键盘上输入 thisisunsafe

注意不要选中按钮或者页面上的什么内容，输入法要在英文状态，然后直接在页面上敲这串字母就行了。

