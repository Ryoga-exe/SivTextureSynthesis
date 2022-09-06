#include "Common.hpp"
 #include "TextureSynthesis.hpp"

void Main() {
    Image input(U"sample/pic1.jpg");
    Texture inputTexture(input);

    Stopwatch sw(StartImmediately::Yes);

    TextureSynthesis synthesis(input, 150, 150, 11);
    DynamicTexture outputTexture;

    synthesis.synthesizeAsync();
    
    while (System::Update()) {

        synthesis.fill(outputTexture);

        PutText(U"input", 40, 15);
        inputTexture.draw(10, 30);

        PutText(U"output", 190, 15);
        outputTexture.draw(160, 30);
    }

    synthesis.exit();
}
