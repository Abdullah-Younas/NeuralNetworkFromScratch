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

unordered_map<char, int> CharacterIndexes;
vector<char> IndexToCharacter;
string book;

void AssigningCharIndexes()
{
    int index = 0;

    for (char character : book)
    {
        if (CharacterIndexes.find(character) == CharacterIndexes.end())
        {
            CharacterIndexes[character] = index;
            IndexToCharacter.push_back(character);

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
    AssigningCharIndexes();

    auto it = CharacterIndexes.find('a');

    if (it != CharacterIndexes.end())
    {
        cout << "Index of a: " << it->second << endl;
    }
    else
    {
        cout << "'a' is not in the training data\n";
    }

    return 0;
}