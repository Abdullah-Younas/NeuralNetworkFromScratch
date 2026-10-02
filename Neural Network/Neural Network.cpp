#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <string>

using namespace std;

unordered_map<char, int> CharacterTokens;
vector<char> TokensToCharacter;
string book;

vector<int> BookTokens;

void TokenizeBook(){
	for (char character : book){
		auto token = CharacterTokens.find(character);

		if (token != CharacterTokens.end())
		{
			BookTokens.push_back(token->second);
		}
	}
}

void AssigningCharTokens()
{
    int index = 0;

    for (char character : book)
    {
        if (CharacterTokens.find(character) == CharacterTokens.end())
        {
            CharacterTokens[character] = index;
            TokensToCharacter.push_back(character);

            index++;
        }
    }
}

void ReadingTrainingText()
{
    ifstream my_file("training.txt");

    if (!my_file.is_open()) {
        cerr << "Error: Could not open the file." << endl;
        return;
    }

    book.assign(
        (istreambuf_iterator<char>(my_file)),
        istreambuf_iterator<char>()
    );

    my_file.close();
}

int main()
{
    ReadingTrainingText();
    AssigningCharTokens();
	TokenizeBook();

	cout << "Vocabulary size: " << CharacterTokens.size() << endl;
    cout << "Book token count: " << BookTokens.size() << endl;

    for (int token : BookTokens)
    {
        cout << token << " ";
    }


    return 0;
}
